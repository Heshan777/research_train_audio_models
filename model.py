import csv
import random

def generate_proposal_dataset(filename="sinhala_call_center_final_1000.csv", num_rows=1000):
    issues = [
        ("Home Broadband router එකේ red light එක blink වෙනවා internet නෑ", "Negative"),
        ("ඊයේ renew කරපු data package එක තාම active වෙලා නෑ zoom meeting එකක් තියෙනවා", "Negative"),
        ("මගේ balance එකෙන් නිකන්ම සල්ලි කැපිලා VAS alert service එකකට refund දාන්න", "Negative"),
        ("Online banking login වෙන්න බෑ account block වෙලා password reset කරගන්න ඕන", "Negative"),
        ("ATM එකෙන් cash ආවෙ නෑ ඒත් account එකෙන් debit වෙලා", "Negative"),
        ("TV connection එකේ signal drop වෙනවා වැස්ස වෙලාවට මේක හදලා දෙන්න", "Neutral"),
        ("Credit card unknown transaction එකක් වෙලා card එක block කරන්න ඉක්මනට", "Negative"),
        ("SIM එක detect වෙන්නෙ නෑ phone එකේ No Service කියලා වැටෙනවා", "Neutral"),
        ("Uber Eats delivery order එක delay වෙලා තාම ලැබුනේ නෑ", "Negative"),
        ("අලුත් Fiber connection එකක් ගන්න packages වල details දැනගන්න ඕන", "Positive"),
        ("Abroad යනවා roaming data pack එකක් active කරගන්න rates දැනගන්න පුළුවන් ද", "Positive"),
        ("Router එක reset කරාට පස්සේ admin password එක වැඩ කරන්නේ නෑ", "Neutral"),
        ("Water board bill එක online pay කරා receipt SMS එක ආවෙ නෑ", "Neutral"),
        ("Prepaid card balance check කරගන්න USSD code එක මොකද්ද", "Positive"),
        ("Credit card annual fee waive off කරගන්න criteria qualify ද බලන්න පුළුවන්ද", "Positive"),
        ("Internet speed එක ගොඩක් slow 1Mbps වත් නෑ unlimited pack එකේ", "Negative")
    ]

    agent_greetings = [
        ("Good morning! Customer care එකෙන්. මම ඔබට සහය වන්නේ කෙසේද?", "Good", "Yes"),
        ("ආයුබෝවන්! Support desk එකෙන්. මම කොහොමද උදව් කරන්නේ?", "Good", "Yes"),
        ("හෙලෝ කියන්න මොකද්ද issue එක?", "Average", "Yes"),
        ("ඔව් කවුද මේ?", "Poor", "No")
    ]

    solutions = {
        "Good": [
            "මම system එකෙන් check කරලා instantly issue එක resolve කරා reference number SMS එකක් දැම්මා.",
            "Technician team එකට high priority escalate කරා අද හවස 4ට කලින් visit එකක් දාලා හදයි.",
            "Account status එක instantly secure කරා dispute form එක email කරලා තියෙන්නේ.",
            "Full refund එක wallet එකට reverse කරා පැය 24ක් ඇතුලත reflect වෙයි."
        ],
        "Average": [
            "මම complaint එක note කරගත්තා system එක update වෙන්න දවස් 2ක් යයි බලන්න.",
            "Router එක off කරලා on කරන්න peak hours වල slow වීම සාමාන්‍ය දෙයක්.",
            "ළඟම තියෙන branch එකට documents අරන් ගිහින් counter එකෙන් check කරගන්න."
        ],
        "Poor": [
            "මට මේකට කරන්න දෙයක් නෑ system එකෙන් error පෙන්නන්නේ.",
            "ඔයාගේ phone එකේ අවුලක් branch එකට ගිහින් බලාගන්න තියන්නම්.",
            "දැනට lines busy පස්සේ call එකක් ගන්න."
        ]
    }

    closings = [
        ("ස්තූතියි අප ඇමතූවාට, සුබ දවසක්!", "Yes"),
        ("Thank you for contacting us, have a wonderful day!", "Yes"),
        ("හරි එහෙනම් තියන්නම්.", "No")
    ]

    # Excel එකේ columns සහ rows නිවැරදිව තබා ගැනීමට utf-8-sig සහ quote_all භාවිතා වේ
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Proposal Table 3 Headers
        writer.writerow([
            "call_id", 
            "audio_file_name", 
            "language_type", 
            "transcript", 
            "sentiment_label", 
            "politeness", 
            "problem_understanding", 
            "solution_quality", 
            "greeting", 
            "closing", 
            "human_qa_score"
        ])

        for i in range(1, num_rows + 1):
            call_id = f"C{i:04d}"
            audio_name = f"call_{i:04d}.wav"
            lang_type = random.choice(["Sinhala", "Sinhala-English"])

            issue_text, raw_sent = random.choice(issues)
            sol_qual = random.choices(["Good", "Average", "Poor"], weights=[0.60, 0.25, 0.15])[0]

            if sol_qual == "Good":
                prob_under = random.choices(["Good", "Average"], weights=[0.85, 0.15])[0]
                politeness = random.choices(["Good", "Average"], weights=[0.85, 0.15])[0]
                sentiment = "Positive" if raw_sent != "Negative" else random.choice(["Positive", "Neutral"])
                greet_txt, _, g_flag = random.choice(agent_greetings[:2])
                close_txt, c_flag = random.choice(closings[:2])
            elif sol_qual == "Average":
                prob_under = random.choices(["Good", "Average", "Poor"], weights=[0.40, 0.50, 0.10])[0]
                polite = random.choices(["Good", "Average", "Poor"], weights=[0.30, 0.50, 0.20])[0]
                politeness = polite
                sentiment = random.choice(["Neutral", "Negative"])
                greet_txt, _, g_flag = agent_greetings[2]
                close_txt, c_flag = random.choice(closings[1:])
            else:
                prob_under = random.choices(["Average", "Poor"], weights=[0.25, 0.75])[0]
                politeness = random.choices(["Average", "Poor"], weights=[0.20, 0.80])[0]
                sentiment = "Negative"
                greet_txt, _, g_flag = random.choice(agent_greetings[3:])
                close_txt, c_flag = closings[2]

            sol_txt = random.choice(solutions[sol_qual])

            # Customer reply based on sentiment
            if sentiment == "Positive":
                cust_reply = "හරි ගොඩක් ස්තූතියි විසඳුමට!"
            elif sentiment == "Neutral":
                cust_reply = "හරි මම බලන්නම් එහෙනම්."
            else:
                cust_reply = "තමුසෙලාගෙන් වැඩක් නෑ හරියට සර්විස් එකක් දෙන්න බෑ."

            # Structured single-line transcript with diarized speaker tags
            transcript = f"Agent: {greet_txt} | Customer: {issue_text} | Agent: {sol_txt} | Customer: {cust_reply} | Agent: {close_txt}"

            # Exact Marks calculation based on Proposal Table 5 Rubric
            score = 0
            if g_flag == "Yes": score += 10
            score += 15 if politeness == "Good" else (10 if politeness == "Average" else 4)
            score += 20 if prob_under == "Good" else (12 if prob_under == "Average" else 5)
            score += 25 if sol_qual == "Good" else (15 if sol_qual == "Average" else 6)
            if c_flag == "Yes": score += 10
            score += 10 if sentiment == "Positive" else (7 if sentiment == "Neutral" else 3)
            score += 10 if politeness == "Good" else (6 if politeness == "Average" else 2) # Acoustic Tone QA

            writer.writerow([
                call_id,
                audio_name,
                lang_type,
                transcript,
                sentiment,
                politeness,
                prob_under,
                sol_qual,
                g_flag,
                c_flag,
                score
            ])

    print(f"Successfully generated {num_rows} records in '{filename}'!")

if __name__ == "__main__":
    generate_proposal_dataset()