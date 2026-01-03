#!/bin/bash

# Model Dashboard Launcher
# Automatically updates registry and starts the web server

echo "🔄 Updating model registry..."
python3 scripts/register_models.py

echo ""
echo "🚀 Starting dashboard server..."
echo "📊 Open your browser to: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

cd model_dashboard && python3 -m http.server 8000
