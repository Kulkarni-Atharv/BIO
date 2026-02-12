#!/usr/bin/env python3
import json
import os
import sys

# Path to local config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "local_config.json")

def main():
    print("=" * 60)
    print("  NETWORK CONFIGURATION WIZARD")
    print("=" * 60)
    print("This tool sets the Laptop IP address without modifying the code.")
    print("It saves settings to data/local_config.json which is ignored by Git.")
    print("-" * 60)
    
    current_config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                current_config = json.load(f)
            print(f"Current Config found: {current_config}")
        except:
            pass
            
    print(f"\nCurrent Laptop IP (MYSQL_HOST): {current_config.get('MYSQL_HOST', 'Not Set')}")
    
    new_ip = input("\nEnter new Laptop IP [Press Enter to keep current]: ").strip()
    
    if new_ip:
        current_config["MYSQL_HOST"] = new_ip
        
        # Ask for password if needed
        pw = input("Enter MySQL Password [Press Enter to keep current]: ").strip()
        if pw:
            current_config["MYSQL_PASSWORD"] = pw
            
        # Save
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        with open(CONFIG_PATH, 'w') as f:
            json.dump(current_config, f, indent=4)
            
        print(f"\n✅ Configuration saved to {CONFIG_PATH}")
        print(f"   MYSQL_HOST = {new_ip}")
    else:
        print("\nNo changes made.")

if __name__ == "__main__":
    main()
