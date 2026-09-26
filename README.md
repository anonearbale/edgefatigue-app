# EdgeFatigue

Upload a raw Apple Watch session recorded with the EduFatigue app and
get a cognitive-fatigue estimate from the trained within-Watch ensemble.
A synthetic demo session is included for visitors without watch data.

## 1. Add the trained models
    cp ~/fatigue_26/watch_models/*.pkl models/

## 2. Pin library versions to the training environment
    source ~/fatigue_26/edufatigue_env/bin/activate
    pip freeze | grep -iE "^(numpy|pandas|scipy|scikit-learn|xgboost|joblib)==" > requirements.txt
    printf "streamlit>=1.50\naltair\n" >> requirements.txt

## 3. Run locally
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

## 4. Deploy to Streamlit Community Cloud
1. Push this folder, including models/, to a GitHub repository.
2. Go to share.streamlit.io and sign in with GitHub.
3. Create app: choose the repository, branch main, file app.py.
4. Under Advanced settings, choose Python 3.10 (the training version).
5. Deploy.

Research prototype. Not a medical device.
