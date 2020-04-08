mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"pennsignals@uphs.upenn.edu\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
baseUrlPath = 'covid_analytics'\n\
" > ~/.streamlit/config.toml
