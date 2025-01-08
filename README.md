# Ai-Customer-Care-Chatbot

## Overview
This project implements an AI-powered customer service chatbot that processes PDF documents and provides real-time, streaming responses to user queries. The bot uses LangGraph for conversation management, Gemini Pro for language processing, and Pinecone for vector storage.

## Features

Real-time streaming responses
PDF document processing and context retrieval
Vector-based semantic search
Asynchronous processing for improved performance
Conversation memory management
Context-aware responses

## Prerequisites

Python 3.8+
Pinecone API key
Google API key (for Gemini Pro)

## Environment Setup

Create a .env file in the project root with the following:

GOOGLE_API_KEY=your_google_api_key

PINECONE_API_KEY=your_pinecone_api_key
