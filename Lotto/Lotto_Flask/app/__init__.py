from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

def create_app():
    app = Flask(__name__)

    # 블루프린트 등록
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # 스케줄러 초기화
    scheduler = BackgroundScheduler()
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

    from .scheduler import fetch_lottery_results
    scheduler.add_job(fetch_lottery_results, 'interval', minutes=1)

    return app
