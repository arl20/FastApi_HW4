from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi_users import FastAPIUsers, models
from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from fastapi.responses import Response, JSONResponse
from fastapi_users.authentication import JWTStrategy
from pydantic import BaseModel
from typing import List
import time
import pandas as pd
import gdown
from dotenv import load_dotenv
import json
from scipy.sparse import csr_matrix, load_npz
import configparser
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Optional
from fastapi import Depends, Request
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from utils import get_link, create_user, load_data, get_key, get_conn
from passlib.context import CryptContext
from aiocache import cached
import aioredis
import logging
import datetime
from functools import partial
import asyncio
import os
from functions_for_recommendation import (games_chosen_to_matrix_line,
                                                get_link,
                                                get_new_recommendations,
                                                get_popular_recommendations,
                                                get_recommendations_by_game,
                                                get_recommendations_by_user,
                                                load_user_data)


async def get_redis():
    host = os.getenv("REDIS_HOST")
    password = os.getenv("REDIS_PASSWORD")
    username = os.getenv("REDIS_USERNAME")
    redis = await aioredis.Redis(host = host, port = 6379, username = username, password = password)
    yield redis
    redis.close()
    
def load_data():
    logging.log(msg="Load games_df", level=logging.INFO)
    df = pd.read_csv('recommendation_bot_data/games_df.csv')
    logging.log(msg="Load matrix", level=logging.INFO)
    matrix = load_npz('recommendation_bot_data/sparse_interaction_matrix.npz')
    logging.log(msg="Load games", level=logging.INFO)
    steam_game_similarities = pd.read_csv('recommendation_bot_data/steam_game_similarities.csv')
    logging.log(msg="Load app_id_to_index", level=logging.INFO)
    with open('recommendation_bot_data/app_id_to_index.json', 'r') as file:
        app_id_to_index = json.load(file)
    logging.log(msg="Load user_id_to_index", level=logging.INFO)         
    with open('recommendation_bot_data/user_id_to_index.json', 'r') as file:
        user_id_to_index = json.load(file)
    logging.log(msg="Loading finished", level=logging.INFO)  
    return df, matrix, app_id_to_index, user_id_to_index, steam_game_similarities


class App(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = configparser.ConfigParser()
        config.read('configs/config.ini')
        self.K_NEIGHBOURS = config.getint('bot', 'K_NEIGHBOURS')
        URL = os.getenv("URL")
        try:
            self.df, self.matrix, self.app_id_to_index, self.user_id_to_index, self.similar_games_df = load_data()
        except FileNotFoundError:
            gdown.download_folder(URL, remaining_ok=True)
            self.df, self.matrix, self.app_id_to_index, self.user_id_to_index, self.similar_games_df = load_data()
        self.games_info_dict_by_name = self.df[~self.df.duplicated(subset=['Name'],
                                                keep='last')].set_index('Name')[['AppID']].to_dict(orient='index')
        self.games_info_dict = self.df.set_index('AppID')[['Name', 'Price',
                                             'Required age', 'About the game',
                                             'Supported languages', 'Genres']].to_dict(orient='index')

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  
load_dotenv()
app = App()

class User(BaseModel):
    user_id: int
    password: str

def check_auth(user: User):
    user_id = user.user_id
    password = user.password
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"select hashed_password from users where id = {user_id}")
    data = cursor.fetchone()
    if data is None:
        return False
    hashed_password = data[0]
    if not pwd_context.verify(user.password, hashed_password):
        return False
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    return True
    
@app.get("/")
async def start():
    info = f"""Cписок функций:<br/>
<b> /register</b> — регистрация<br/>
<b> /get_list_of_all_game</b> - получить список всех игр (либо игр, название которых содержит определенную по><b> /add_game</b> — добавить игру к списку для рекомендаций<br/>
<b> /delete_game</b> — удалить игру из списка для рекомендаций<br/>
<b> /info_game</b> — информация о конкретной игре<br/>
<b> /similar_games</b> - рекомендация игр, похожих на конкретную игру<br/>
<b> /get_list_of_game</b> — получение текущего списка любимых игр<br/>
<b> /clear_list_of_game</b> — очистка списка любимых игр<br/>
<b> /set_k</b> — установить значение количества рекомендуемых игр<br/>
<b>/similar_games</b> —  порекомендовать игры, похожие на конкретную игру<br/>
<b>/get_recommended_game</b> - получение игр для рекомендации<br/>
<b> /post_review</b> — оставить отзыв о работе сервиса<br/>
"""
    return Response(content=info, status_code=200, media_type="text/html")
  

@app.post("/register")
async def register(user: User, k_games: int = Query(10, description="Количество игр, которое будет порекомендовано пользователю")):
    """
    Регистрация пользователя
    """
    hashed_password = pwd_context.hash(user.password)
    res = create_user(user.user_id, k_games, hashed_password)
    if res:
        return {"message" : f"Пользователь c id {user.user_id} зарегистрирован"}
    else:
        raise HTTPException(status_code=422, detail=f"Пользователь с user_id {user.user_id} уже существует")
        
@app.get("/get_list_of_all_game")
async def get_list_of_all_game(query: str = Query(None, description="Необязательный параметр. \
Если он заполнен, то возвращаются только игры, \
которые содержат конкретную подстроку.")):
    """
    Получение списка всех игр из базы.\n
    """
    if query is None:
        return {"list" : list(app.games_info_dict_by_name.keys())}
    else:
        return {"list" : [game for game in app.games_info_dict_by_name if query.lower() in game.lower()]}
    
@app.post("/add_game")
async def add_game(current_user: User,
                   game: str = Query(description="Название игры из списка игр")):
    """
    Добавление игры в список любимых игр пользователя 
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    try:
        appid = app.games_info_dict_by_name[game]['AppID']
    except KeyError:
        raise HTTPException(status_code=422, detail="Данная игра отсутствует в базе")
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"select list_games, hashed_password from users where id = {user_id}")
    games, hashed_password = cursor.fetchone()
    if not pwd_context.verify(current_user.password, hashed_password):
        raise HTTPException(status_code=401, detail="Authentication failed")
    if len(games) > 0:
        games = set(games)
    else:
        games = set()
    games.add(appid)
    cursor.execute(f"UPDATE users SET list_games = ARRAY{str(sorted(games))}::integer[] WHERE id = {user_id}")
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    
    logging.log(msg=f"{game} add to {user_id} list", level=logging.INFO)
    info = f"Игра {game} добавлена в список ваших любимых игр"
    return Response(content=info, status_code=200, media_type="text/html")


@app.post("/delete_game")
async def delete_game(current_user: User,
                      game = Query(description="Название игры из списка игр")):
    """
    Удаление игры из списка любимых игр пользователя 
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    try:
        appid = app.games_info_dict_by_name[game]['AppID']
    except KeyError:
        raise HTTPException(status_code=422, detail="Игра не найдена")
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"select list_games from users where id = {user_id}")
    games = set(cursor.fetchone()[0])
    try:
        games.remove(appid)
        cursor.execute(f"UPDATE users SET list_games = ARRAY{str(sorted(games))}::integer[] WHERE id = {user_id}")
        info = f"Игра {game} удалена из списка ваших любимых игр"     
    except (KeyError, ValueError):
        info = "Данной игры нет в вашем списке"
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    return Response(content=info, status_code=200, media_type="text/html")
    
@app.get("/info_game")
async def info_game(game: str = Query(description="Название игры из списка игр")):
    """
    Информация о конкретной игре
    """
    try:
        appid = app.games_info_dict_by_name[game]['AppID']
        info = f"""Информация об игре {app.games_info_dict[appid]['Name']}
<b>Возрастное ограничение:</b> {app.games_info_dict[appid]['Required age']}
<b>Жанры:</b> {app.games_info_dict[appid]['Genres']}
<b>Описание:</b> {app.games_info_dict[appid]['About the game']}
<a href="store.steampowered.com/app/{appid}">Ссылка на steam</a>
"""
        return Response(content=info, status_code=200, media_type="text/html")
    except KeyError:
        raise HTTPException(status_code=422, detail="Игра не найдена")
    
@app.post("/similar_games")
async def similar_games(k_games: int = Query(description="Количество игр"),
                        game = Query(description="Название игры из списка игр"),
                        redis: aioredis.Redis = Depends(get_redis)):
    """
    Получение списка игр, похожих на конкретную игру
    """
    try:
        appid = app.games_info_dict_by_name[game]['AppID']
    except KeyError:
        return {"message": f"Игра {game} не найдена"}
    cached_info = await redis.get(f"{appid}:{k_games}:similar_games")
    if cached_info:
        return Response(content=cached_info.decode(), status_code=200, media_type="text/html")
    recommendations_by_game = await asyncio.to_thread(get_recommendations_by_game,
                                                      app.similar_games_df,
                                                      [appid], k_games)
    get_game_link = partial(get_link, games_info_dict=app.games_info_dict)
    recommendations_by_game_answer = '\n'.join(map(get_game_link, recommendations_by_game))
    info = f"""<b> Игры, похожие на игру {game}:</b>
{recommendations_by_game_answer}
"""
    await redis.set(f"{appid}:{k_games}:similar_games", info)
    await redis.expire(f"{appid}:{k_games}:similar_games", 3600)
    return Response(content=info, status_code=200, media_type="text/html")
    
@app.post("/set_k")
async def set_k_games(current_user: User, k: int):
    """
    Установка количества игр k для пользователя
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    try:
        if k > 0 and k <= 20:
            info = f'Установлено значение количества рекомендованных игр {k}'
            tunnel, conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(f"select k_games from users where id = {user_id}")
            k_games = cursor.fetchone()
            if len(k_games) == 0:
                create_user(user_id, k)
            else:
                cursor.execute(f"UPDATE users SET k_games = {k} WHERE id = {user_id}")
            cursor.close()
            conn.commit()
            conn.close()
            tunnel.stop()
            return {"message": info}
        else:
            info = "k должно быть целым положительным числом от 1 до 20"
            return {"message": info}
    except (AttributeError, ValueError):
        info = "k должно быть целым положительным числом от 1 до 20"
    return Response(content=info, status_code=200, media_type="text/html")
        
@app.post("/get_list_of_game")
async def get_list_of_game(current_user: User):
    """
    Получение списка любимых игр пользователя
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"select list_games from users where id = {user_id}")
    games = cursor.fetchone()[0]
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    if len(games) == 0:
        info = 'Список ваших любимых игр пуст\n'
    else:
        info = 'Текущий список ваших любимых игр:\n' + '\n'.join(sorted(map(lambda x: app.games_info_dict[x]['Name'], games)))
    return Response(content=info, status_code=200, media_type="text/html")
    
@app.post("/clear_list_of_game")
async def clear_list_of_game(current_user: User):
    """
    Очистка списка любимых игр пользователя
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE users SET list_games = ARRAY[]::integer[] WHERE id = {user_id}")
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    info = 'Список ваших любимых игр очищен'
    return Response(content=info, status_code=200, media_type="text/html")

@app.post("/get_recommended_game")
async def get_recommended_game(current_user: User, redis: aioredis.Redis = Depends(get_redis)):
    """
    Получение рекомендации игр (ML-часть)
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_id = current_user.user_id
    cached_last_list = await redis.get(f"user:{user_id}:list_of_games_before_last_recommendation")
    cached_k = await redis.get(f"user:{user_id}:k_before_last_recommendation")
    cached_recommendation = await redis.get(f"user:{user_id}:recommended_game")
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"select k_games, list_games, hashed_password from users where id = {user_id}")
    data = cursor.fetchone()
    if data is None:
        raise HTTPException(status_code=401, detail="Authentication failed")
    k_games, list_games, hashed_password = data
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    if cached_recommendation and cached_last_list and cached_k \
    and cached_last_list.decode() ==  ' '.join(map(str, sorted(set(list_games)))) \
    and int(cached_k.decode()) == k_games:
        return Response(content=cached_recommendation.decode(), status_code=200, media_type="text/html")        
    get_game_link = partial(get_link, games_info_dict=app.games_info_dict)
    popular_list = await asyncio.to_thread(get_popular_recommendations, app.df, k_games)
    popular_list_answer = '\n'.join(map(get_game_link, popular_list))
    new_list = await asyncio.to_thread(get_new_recommendations, app.df, k_games)
    new_list_answer = '\n'.join(map(get_game_link, new_list))
    if len(list_games) == 0:
        info=f"""К сожалению, вы не дали мне информации
 о ваших любимых играх, поэтому я не могу сделать персональную рекомендацию. Но вы можете поиграть в самые популярные игры!
<b>Популярные игры с высоким рейтингом:</b>
{popular_list_answer}\n
<b>Набирающие популярность новинки:</b>
{new_list_answer}
        """
        return Response(content=info, status_code=200, media_type="text/html")

    recommendations_by_game = await asyncio.to_thread(get_recommendations_by_game,
                                                      app.similar_games_df,
                                                      list_games, k_games)
    user_row = await asyncio.to_thread(games_chosen_to_matrix_line, list_games, app.df, app.app_id_to_index)
    recommendations_by_user = await asyncio.to_thread(get_recommendations_by_user, app.matrix,
                                                      user_row, app.K_NEIGHBOURS, k_games,
                                                      app.user_id_to_index, app.app_id_to_index, False)
    recommendations_by_game_answer = '\n'.join(map(get_game_link, recommendations_by_game))
    recommendations_by_user_answer = '\n'.join(map(get_game_link, recommendations_by_user))
    info = f"""<b>Пользователи, похожие на Вас, играют в:</b>
{recommendations_by_user_answer}\n
<b>Игры, похожие на те, что Вы играли:</b>
{recommendations_by_game_answer}\n
<b>Популярные игры с высоким рейтингом:</b>
{popular_list_answer}\n
<b>Набирающие популярность новинки:</b>
{new_list_answer}"""
    await redis.set(f"user:{user_id}:recommended_game", info)
    await redis.set(f"user:{user_id}:list_of_games_before_last_recommendation", ' '.join(map(str, sorted(set(list_games)))))
    await redis.set(f"user:{user_id}:k_before_last_recommendation", k_games)
    return Response(content=info, status_code=200, media_type="text/html")

@app.post("/post_review")
async def post_review(current_user: User, 
                      score: int = Query(description="Оценка от 1 до 5"), 
                      review: str = Query(description="Отзыв")):
    """
    Отправка отзыва о сервисе
    """
    if not check_auth(current_user):
        raise HTTPException(status_code=401, detail="Authentication failed")
    timestamp = str(datetime.datetime.fromtimestamp(time.time()))
    user_id = current_user.user_id
    tunnel, conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""INSERT INTO review_history (user_id, datetime, mark, review)
                       VALUES ({user_id}, '{timestamp}', {score},'{review}')""")
    cursor.close()
    conn.commit()
    conn.close()
    tunnel.stop()
    return Response(content=f"Спасибо за отзыв!\n Ваш отзыв: {review}", status_code=200, media_type="text/html")
