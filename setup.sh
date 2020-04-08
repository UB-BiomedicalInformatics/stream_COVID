mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"pennsignals@uphs.upenn.edu\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=true\n\
port = $PORT\n\
baseUrlPath = 'covid_analytics'\n\
serverAddress = 'https://halsted.compbio.buffalo.edu'\n\
" > ~/.streamlit/config.toml
