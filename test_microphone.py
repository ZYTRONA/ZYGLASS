#!/usr/bin/env python3
"""
Test microphone detection and audio input for ZYGLASS Voice Assistant.
Run this to verify your AirPods or other microphone is detected and working.
"""
import sounddevice as sd
import numpy as np

def list_audio_devices():
    """List all available audio input devices."""
    print("=" * 60)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 60)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            marker = "  "
            name_lower = device['name'].lower()
            
            # Highlight likely devices
            if any(keyword in name_lower for keyword in ['airpod', 'bluetooth', 'headset', 'wireless']):
                marker = "🎧"
            elif i == sd.default.device[0]:
                marker = "⭐"
            
            print(f"\n{marker} Device {i}: {device['name']}")
            print(f"   Channels: {device['max_input_channels']} input")
            print(f"   Sample Rate: {device['default_samplerate']} Hz")
    
    print("\n" + "=" * 60)
    print(f"Default input device: {sd.default.device[0]}")
    print("=" * 60)
    
    return input_devices

def test_microphone(device_id=None):
    """Test microphone by recording a short sample and showing volume levels."""
    print("\n" + "=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)
    
    if device_id is None:
        device_id = sd.default.device[0]
    
    device_info = sd.query_devices(device_id, 'input')
    print(f"Testing device: {device_info['name']}")
    print("\nSpeak into your microphone for 5 seconds...")
    print("You should see volume bars if audio is detected:\n")
    
    def callback(indata, frames, time, status):
        """Show real-time volume level."""
        volume_norm = np.linalg.norm(indata) * 10
        bar_length = int(min(volume_norm, 50))
        bar = '█' * bar_length
        print(f"\r[{bar:<50}] {volume_norm:5.1f}", end='', flush=True)
    
    try:
        with sd.InputStream(device=device_id, channels=1, callback=callback, samplerate=16000):
            sd.sleep(5000)  # 5 seconds
        
        print("\n\n✓ Microphone test complete!")
        print("If you saw volume bars, your microphone is working.")
        print("If no bars appeared, check:")
        print("  1. Microphone is not muted")
        print("  2. AirPods are connected and selected as input device")
        print("  3. System audio permissions are granted")
        
    except Exception as e:
        print(f"\n✗ Error testing microphone: {e}")
        print("\nTry selecting a different device with: python test_microphone.py <device_id>")

def main():
    """Main test function."""
    import sys
    
    # List all devices
    input_devices = list_audio_devices()
    
    # Test specific device or default
    if len(sys.argv) > 1:
        device_id = int(sys.argv[1])
        print(f"\nTesting device {device_id} as specified...")
    else:
        # Try to find AirPods or Bluetooth device
        device_id = None
        for dev_id, dev_info in input_devices:
            name_lower = dev_info['name'].lower()
            if any(keyword in name_lower for keyword in ['airpod', 'bluetooth', 'headset', 'wireless']):
                device_id = dev_id
                print(f"\n🎧 Found wireless device: {dev_info['name']}")
                print(f"   Using device {device_id} for test...")
                break
        
        if device_id is None:
            print("\nNo wireless headset detected, using default input...")
    
    test_microphone(device_id)
    
    print("\n" + "=" * 60)
    print("TIP: To use a specific device, run:")
    print("  python test_microphone.py <device_id>")
    print("=" * 60)

if __name__ == "__main__":
    main()
