#!/usr/bin/env python3
"""
Windows Emoji Fix - Replace problematic emojis with Windows-safe text
"""

emoji_replacements = {
    "🚀": "[ROCKET]",
    "🎯": "[TARGET]", 
    "📊": "[CHART]",
    "💡": "[IDEA]",
    "🏦": "[BANK]",
    "💰": "[MONEY]",
    "📡": "[SIGNAL]",
    "❌": "[ERROR]",
    "✅": "[SUCCESS]",
    "😴": "[IDLE]",
    "🏆": "[WINNER]",
    "⚡": "[TURBO]",
    "🔄": "[CYCLE]",
    "🛑": "[STOP]",
    "📈": "[UP]",
    "🕒": "[TIME]",
    "💼": "[PORTFOLIO]",
    "🧠": "[BRAIN]",
    "💾": "[SAVE]",
    "🐌": "[SLOW]",
    "🔥": "[FIRE]"
}

def fix_windows_emojis(file_path):
    """Replace problematic emojis with Windows-safe text"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FIXED] Updated {file_path} with Windows-safe logging")
            return True
        else:
            print(f"[SKIP] No emojis found in {file_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to fix {file_path}: {e}")
        return False

if __name__ == "__main__":
    files_to_fix = [
        "alpaca_paper_trading_maximal.py"
    ]
    
    print("[WINDOWS FIX] Starting emoji replacement for Windows compatibility...")
    
    for file_path in files_to_fix:
        if fix_windows_emojis(file_path):
            print(f"[SUCCESS] Fixed {file_path}")
        
    print("[COMPLETE] Windows emoji fix completed!")