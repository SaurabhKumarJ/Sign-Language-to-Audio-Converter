import pyttsx3

def speak_sign(sign):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Convert the sign string to speech
    engine.say(sign)
    
    # Wait for the speech to complete
    engine.runAndWait()

# Example usage:
speak_sign("Hello, how are you?")
