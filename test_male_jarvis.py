#!/usr/bin/env python3
"""
Test script to verify Jarvis uses male voice
"""
import os
import pyttsx3

def test_male_voice():
    """Test that Jarvis uses a male voice"""
    print("🎤 Testing Jarvis male voice...")
    
    # Initialize TTS engine
    try:
        engine = pyttsx3.init(driverName="sapi5")  # Force Windows SAPI
    except Exception:
        engine = pyttsx3.init()
    
    # Get current voice
    current_voice = engine.getProperty("voice")
    voices = engine.getProperty("voices")
    
    print(f"Current voice ID: {current_voice}")
    
    # Find the current voice details
    current_voice_name = "Unknown"
    for voice in voices:
        if voice.id == current_voice:
            current_voice_name = voice.name
            break
    
    print(f"Current voice name: {current_voice_name}")
    
    # List all available voices
    print("\nAvailable voices:")
    for i, voice in enumerate(voices):
        gender = "Male" if "david" in voice.name.lower() or "james" in voice.name.lower() or "mark" in voice.name.lower() else "Female"
        marker = "🎯" if voice.id == current_voice else "  "
        print(f"{marker} {i}: {voice.name} ({voice.id}) - {gender}")
    
    # Test speech
    print("\n🗣️  Testing speech with current voice...")
    test_text = "Hello, I am Jarvis, your AI assistant. How may I help you today?"
    engine.say(test_text)
    engine.runAndWait()
    
    print("\n✅ Voice test complete!")

if __name__ == "__main__":
    test_male_voice()

