import google.generativeai as genai

genai.configure(api_key="AIzaSyCErMko7fBRNH4MMmdGvKimnBbgeISh7Bc")

model = genai.GenerativeModel("models/gemini-1.5-flash")

response = model.generate_content("Explain how diabetes is detected.")
print(response.text)
