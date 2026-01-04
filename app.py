from flask import Flask, render_template, request, send_file
import joblib, sqlite3, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

RECORDS_SECRET_KEY = "heart123"

model = joblib.load("heart_disease_model.pkl")
features = joblib.load("model_features.pkl")

def init_db():
    conn = sqlite3.connect("heart_data.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS patient_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, phone TEXT, gender TEXT,
        age INTEGER, cp INTEGER, trestbps INTEGER,
        chol INTEGER, thalach INTEGER, oldpeak REAL,
        prediction TEXT, probability REAL, risk_level TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/", methods=["GET","POST"])
def home():
    result=risk=prob=None

    if request.method=="POST":
        d=request.form
        sex=1 if d["gender"]=="Male" else 0

        X=pd.DataFrame([[int(d["age"]),sex,int(d["cp"]),
            int(d["trestbps"]),int(d["chol"]),
            int(d["thalach"]),float(d["oldpeak"])]],
            columns=features)

        p=model.predict_proba(X)[0][1]
        prob=round(p*100,2)
        result="YES" if p>=0.45 else "NO"

        if p<0.4:risk="LOW RISK"
        elif p<0.7:risk="MODERATE RISK"
        else:risk="HIGH RISK"

        conn=sqlite3.connect("heart_data.db")
        c=conn.cursor()
        c.execute("""INSERT INTO patient_data
        (name,phone,gender,age,cp,trestbps,chol,thalach,oldpeak,
         prediction,probability,risk_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (d["name"],d["phone"],d["gender"],d["age"],d["cp"],
         d["trestbps"],d["chol"],d["thalach"],d["oldpeak"],
         result,prob,risk))
        conn.commit();conn.close()

    return render_template("index.html",result=result,risk=risk,prob=prob)

@app.route("/records",methods=["GET","POST"])
def records():
    if request.method=="POST":
        if request.form.get("key")!=RECORDS_SECRET_KEY:
            return render_template("records_key.html",error="Invalid Key")

        conn=sqlite3.connect("heart_data.db")
        data=conn.execute("SELECT * FROM patient_data").fetchall()
        conn.close()
        return render_template("records.html",data=data)

    return render_template("records_key.html")

@app.route("/ai-analysis")
def ai_analysis():
    conn=sqlite3.connect("heart_data.db")
    df=pd.read_sql("SELECT * FROM patient_data",conn)
    conn.close()

    low=mod=high=0
    summary="No data available."

    if not df.empty:
        low=len(df[df.risk_level=="LOW RISK"])
        mod=len(df[df.risk_level=="MODERATE RISK"])
        high=len(df[df.risk_level=="HIGH RISK"])
        summary=f"We analyzed {len(df)} patients. {high} high risk, {mod} moderate risk, {low} low risk."

    return render_template("ai_analysis.html",
        summary=summary,
        labels=["Low","Moderate","High"],
        values=[low,mod,high])

@app.route("/pdf/<int:pid>")
def pdf(pid):
    conn=sqlite3.connect("heart_data.db")
    p=conn.execute("SELECT * FROM patient_data WHERE id=?", (pid,)).fetchone()
    conn.close()

    file=f"patient_{pid}.pdf"
    c=canvas.Canvas(file,pagesize=A4)
    c.drawString(100,800,"Heart Disease AI Report")
    c.drawString(100,760,f"Name: {p[1]}")
    c.drawString(100,740,f"Age: {p[4]}")
    c.drawString(100,720,f"Cholesterol: {p[7]}")
    c.drawString(100,700,f"Risk Level: {p[11]}")
    c.save()
    return send_file(file,as_attachment=True)

if __name__=="__main__":
    app.run(debug=True)
