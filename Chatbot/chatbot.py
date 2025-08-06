import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)

class AdmissionChatBot:
    def __init__(self):
        self.ps = PorterStemmer()
        self.load_knowledge_base("knowledge_base.json")
        self.context = {}

        # Original keywords (unstemmed)
        self.intent_keywords_raw = {
            "deadlines": ["deadlin", "last", "date", "close", "due"],
            "eligibility": ["eligib", "qualif", "criteria", "require"],
            "fees": ["fee", "tuit", "cost", "price", "charg", "amount", "₹", "rs"],
            "contact": ["contact", "email", "phone", "reach", "helpdesk"],
            "programs": ["program", "course", "mca", "special", "subject", "field"],
            "installments": ["install", "payment", "emi", "split", "part", "plan"],
            "hostel": ["hostel", "accommodation", "stay", "room", "residence"],
            "placements": ["placement", "job", "career", "opportunit", "recruit"],
            "scholarships": ["scholar", "aid", "grant", "financial"],
            "documents": ["document", "certificat", "mark", "photo", "proof", "id"],
            "duration": ["duration", "length", "year", "time", "semest"],
            "syllabus": ["syllabus", "subject", "curriculum", "topic", "paper"],
            "admission_process": ["apply", "process", "how", "steps", "procedure", "admission"],
            "mode_of_admission": ["mode", "entrance", "exam", "merit", "direct", "admission"],
            "seat_availability": ["seat", "availability", "vacancy", "number", "intake"],
            "faculty": ["faculty", "professor", "teacher", "staff", "lecturer", "mentor"],
            "campus_facilities": ["campus", "facility", "lab", "library", "sports", "wifi","internet", "cafeteria"],
            "internship": ["internship", "intern", "training", "project", "work"]
        }

        # Stem all keywords once and save
        self.intent_keywords = {}
        for intent, keywords in self.intent_keywords_raw.items():
            stemmed_keywords = set(self.ps.stem(word) for word in keywords)
            self.intent_keywords[intent] = stemmed_keywords

    def load_knowledge_base(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            self.knowledge = json.load(f)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return [self.ps.stem(t) for t in tokens if t.isalnum()]

    def match_intent(self, query):
        tokens = set(self.preprocess(query))
        for intent, keywords in self.intent_keywords.items():
            if tokens.intersection(keywords):
                return intent
        return None

    def generate_response(self, intent):
        if intent in self.knowledge:
            return self.knowledge[intent]
        return ("Sorry, I didn't understand that. "
                "You can ask me about fees, eligibility, deadlines, programs, contact info, or installments.")

    def handle_query(self, query):
        intent = self.match_intent(query)
        print(f"Detected intent: {intent}")
        response = self.generate_response(intent)
        self.log_conversation(query, response, intent)
        return response

    def log_conversation(self, query, response, intent):
        with open("conversations_log.txt", "a", encoding='utf-8') as f:
            f.write(f"User: {query}\nIntent: {intent}\nBot: {response}\n\n")


knowledge_base = {
    "deadlines": "MCA admissions close on September 30, 2025. Late applications are accepted until October 15 with a late fee.",
    "eligibility": "Eligibility: Bachelor's degree with at least 50% marks and a valid entrance exam score.",
    "fees": "The total tuition fee is ₹1,20,000 per year. Scholarships are offered to the top 10% scorers.",
    "contact": "Email: mcaadmissions@mca | Phone: +91-0123456789",
    "programs": "We offer MCA programs with specializations in Artificial Intelligence, Cybersecurity, and Data Science.",
    "installments": "You will be given 2 installments to pay the yearly fee.",
    "hostel": "Yes, hostel accommodation is available for both boys and girls with mess facilities and 24/7 security.",
    "placements": "The placement cell has partnerships with top IT companies. 85% of students get placed each year.",
    "scholarships": "Merit-based scholarships are available. Students scoring above 85% in UG can apply.",
    "documents": "You need to submit UG mark sheets, entrance scorecard, passport photo, ID proof, and transfer certificate.",
    "duration": "The MCA program is a 2-year full-time course divided into 4 semesters.",
    "syllabus": "The syllabus includes Programming in Python, Data Structures, Mathematics, Operating Systems, Object Oriented Programming.",
    "admission_process": "To apply, fill the online application form, upload required documents, pay the application fee, and appear for the entrance exam.",
    "mode_of_admission": "Admissions are through a valid entrance exam score or based on merit in qualifying exams as per university rules.",
    "seat_availability": "There are 60 seats available in the MCA program each academic year.",
    "faculty": "Our faculty includes experienced professors and industry experts dedicated to quality education.",
    "campus_facilities": "Campus facilities include well-equipped computer labs, library, sports grounds, Wi-Fi, and cafeteria.",
    "internship": "Students get internship opportunities with partner IT companies during the summer semester."
}

if __name__ == "__main__":
    # Write knowledge base JSON file (overwrite existing)
    with open("knowledge_base.json", "w", encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    bot = AdmissionChatBot()
    print("Admission Bot: Hi! Ask me about MCA admissions (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Thank you! Feel free to reach out for more help.")
            break
        print("Bot:", bot.handle_query(user_input))
