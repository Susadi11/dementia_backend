"""
Generate test audio file with Text-to-Speech for reminder testing
"""

from gtts import gTTS
import os

# Create output directory if it doesn't exist
os.makedirs("data/sample/audio", exist_ok=True)

# Test phrases for reminders
test_phrases = {
    "reminder_medicine.wav": "Remind me to take my blood pressure medicine at 8 AM every morning",
    "reminder_appointment.wav": "Set a reminder for doctor appointment next Tuesday at 2 PM",
    "reminder_lunch.wav": "I need to remember my lunch at noon daily",
    "reminder_pills.wav": "Remind me to take my pills at 9 PM tonight",
}

print("🎤 Generating test audio files with speech...\n")

for filename, text in test_phrases.items():
    output_path = f"data/sample/audio/{filename}"
    
    try:
        # Create speech
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        tts.save(output_path)
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Created: {filename}")
        print(f"   Text: '{text}'")
        print(f"   Size: {file_size:,} bytes\n")
        
    except Exception as e:
        print(f"❌ Failed to create {filename}: {e}\n")

print("✅ Audio generation complete!")
print("\nYou can now test with:")
print("python test_voice_reminder.py data/sample/audio/reminder_medicine.wav patient_001 high")
