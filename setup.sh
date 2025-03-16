#!/bin/bash

# BidPrice Prediction Model Setup Script

echo "🚀 Setting up BidPrice prediction model project..."

# Python 버전 확인
echo "🔍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
if [ "$python_version" != "3.11.11" ]; then
    echo "❌ Python 3.11.11 required, but found $python_version. Please install Python 3.11.11."
    exit 1
fi
echo "✅ Python 3.11.11 detected."

# 필요한 디렉토리 생성
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs
mkdir -p src
mkdir -p tests