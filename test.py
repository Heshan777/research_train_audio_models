import joblib

# 1. Load trained pipelines
models = {
    "sentiment": joblib.load("trained_models/sentiment_label_pipeline.joblib"),
    "politeness": joblib.load("trained_models/politeness_pipeline.joblib"),
    "understanding": joblib.load("trained_models/problem_understanding_pipeline.joblib"),
    "solution": joblib.load("trained_models/solution_quality_pipeline.joblib")
}

# 2. Sample Unseen Call Transcript (Incoming from STT)
raw_transcript = (
    "Agent: Good morning! Customer care එකෙන්. මම ඔබට සහය වන්නේ කෙසේද? | "
    "Customer: මගේ Home Broadband router එකේ red light එකක් blink වෙනවා ඊයේ ඉඳන්. Internet වැඩ නෑ. | "
    "Agent: හරි සර්, මම system එකෙන් check කරලා line එක reset කරා. Reference number එකක් SMS කරා. | "
    "Customer: හරි ගොඩක් ස්තූතියි විසඳුමට! | "
    "Agent: Thank you for calling us, have a great day!"
)

# Extract Speaker Texts
parts = raw_transcript.split("|")
cust_text = " ".join([p.replace("Customer:", "").strip() for p in parts if "Customer:" in p])
agent_text = " ".join([p.replace("Agent:", "").strip() for p in parts if "Agent:" in p])

# 3. Model Inference
pred_sentiment = models["sentiment"].predict([cust_text])[0]
pred_politeness = models["politeness"].predict([agent_text])[0]
pred_understanding = models["understanding"].predict([agent_text])[0]
pred_solution = models["solution"].predict([agent_text])[0]

# 4. Rubric Scoring (Proposal Table 5)
score_greet = 10 if any(w in agent_text.lower() for w in ["good morning", "good afternoon", "ආයුබෝවන්"]) else 0
score_polite = 14 if pred_politeness == "Good" else (10 if pred_politeness == "Average" else 4)
score_under = 18 if pred_understanding == "Good" else (12 if pred_understanding == "Average" else 5)
score_sol = 22 if pred_solution == "Good" else (15 if pred_solution == "Average" else 6)
score_close = 10 if any(w in agent_text.lower() for w in ["thank you", "ස්තූතියි", "great day", "සුබ දවසක්"]) else 0
score_sent_handle = 9 if pred_sentiment == "Positive" else (7 if pred_sentiment == "Neutral" else 4)
score_tone = 9 if pred_politeness == "Good" else 6

final_qa_score = score_greet + score_polite + score_under + score_sol + score_close + score_sent_handle + score_tone

# 5. Build Explanation Points
strengths, improvements = [], []

if score_greet == 10:
    strengths.append("Agent greeted the customer properly.")
else:
    improvements.append("Agent missed the opening greeting.")

if pred_politeness == "Good":
    strengths.append("Agent maintained polite communication.")
else:
    improvements.append("Agent politeness needs improvement.")

if pred_understanding == "Good":
    strengths.append("Customer's problem was understood clearly.")
else:
    improvements.append("Problem understanding was suboptimal.")

if pred_solution == "Good":
    strengths.append("Provided a complete and reliable solution.")
else:
    improvements.append("The solution could have been explained more clearly.")

if pred_sentiment == "Negative":
    improvements.append("Customer sentiment remained negative during the call.")

# 6. Render Output
formatted_transcript = raw_transcript.replace(" | ", "\n")

print(f"""
CALL ID: CALL_001

────────────────────────────────
1. TRANSCRIPT
────────────────────────────────
{formatted_transcript}

────────────────────────────────
2. CUSTOMER SENTIMENT
────────────────────────────────
Sentiment: {pred_sentiment}

────────────────────────────────
3. AGENT TONE & SPEAKING QUALITY
────────────────────────────────
Tone: {'Calm' if pred_politeness != 'Poor' else 'Rude'}
Speaking Quality: {pred_politeness}

────────────────────────────────
4. QA EVALUATION
────────────────────────────────
Greeting                 : {score_greet}/10
Politeness               : {score_polite}/15
Problem Understanding    : {score_under}/20
Solution Quality         : {score_sol}/25
Closing                  : {score_close}/10
Sentiment Handling       : {score_sent_handle}/10
Tone & Speaking Quality  : {score_tone}/10

────────────────────────────────
5. FINAL QA SCORE
────────────────────────────────
Score: {final_qa_score}/100

────────────────────────────────
6. EXPLANATION
────────────────────────────────
Strengths:
""" + "\n".join([f"• {s}" for s in strengths]) + "\n\nAreas for Improvement:\n" + "\n".join([f"• {i}" for i in improvements]) + f"""

Overall:
The agent achieved a score of {final_qa_score}/100.
""")