#!/bin/bash
echo "Setting up SoundInk 24/7 Background Service..."

# مسارات
PLIST_PATH="$HOME/Library/LaunchAgents/com.salim.soundink.plist"
PROJECT_DIR="/Users/alyasi/apva"
LOG_OUT="$PROJECT_DIR/logs/web_server.log"
LOG_ERR="$PROJECT_DIR/logs/web_server_err.log"

# إنشاء مجلد السجلات إذا لم يكن موجوداً
mkdir -p "$PROJECT_DIR/logs"

# إنشاء ملف الخدمة الخاص بنظام الماك
cat << EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.salim.soundink</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>export PATH="/opt/homebrew/Caskroom/miniforge/base/envs/apva310/bin:/opt/homebrew/bin:/usr/local/bin:$PATH" && cd "$PROJECT_DIR" && python app.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StandardOutPath</key>
    <string>$LOG_OUT</string>
    <key>StandardErrorPath</key>
    <string>$LOG_ERR</string>
</dict>
</plist>
EOF

# إيقاف الخدمة لو كانت تعمل مسبقًا
launchctl unload "$PLIST_PATH" 2>/dev/null

# تشغيل الخدمة
launchctl load -w "$PLIST_PATH"
launchctl start com.salim.soundink

echo "✅ تم تثبيت وتشغيل الخدمة بنجاح!"
echo "خادم SoundInk يعمل الآن في الخلفية 24/7."
echo "الرابط دائماً متاح عبر: http://localhost:5000"
echo ""
echo "لإيقاف الخدمة متى ما شئت، استخدم الأمر:"
echo "launchctl unload $PLIST_PATH"
