model.py is the backend part. Run only model.py 

To run it.
1. First make environment typing following commands on terminal

python -m venv env
env/Scripts/activate

2.Create a .env file in the directory you're working with and add GOOGLE_API_KEY = "Your_API_key"

3. Then add dependencies

pip install -r requirements.txt
43.  run server

uvicorn model:app --reload

5. do to the swagger window

type on browser

http://127.0.0.1:8000/docs








rag_for_pdf.py is used to create db . Dont run it, because it will change our vector db
