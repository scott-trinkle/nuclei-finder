
echo "[general]" > ~/.streamlit/credentials.toml
echo "email = \"tscott.trinkle@gmail.com\"" >> ~/.streamlit/credentials.toml

echo "[server]" > ~/.streamlit/config.toml
echo "headless = True" >> ~/.streamlit/config.toml
echo "enableCORS=false" >> ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml

echo "[theme]" >> ~/.streamlit/config.toml
echo "base=\"light\"" >> ~/.streamlit/config.toml
echo "primaryColor=\"#800000\"" >> ~/.streamlit/config.toml
echo "secondaryBackgroundColor=\"#d6d6ce\"" >> ~/.streamlit/config.toml
