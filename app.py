# 必要なライブラリをインポートします
#import builtins
import openai
import streamlit as st

# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
openai.api_key = st.secrets.OpenAIAPI.openai_api_key    

# ユーザーの入力テキストから埋め込みベクトルを作成する関数を定義します
def createEmbedding(user_text):
    try:
        response = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=user_text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(e)
        raise e

# 事前に用意した知識のベクトルを作成する関数を定義します
def createChishikiVector():
    chishiki = [
        '株式会社インプレスの設立は 1995年10月16日 です。',
        '株式会社インプレスの資本金は 5,000万円 です。',
        '株式会社インプレスの代表者は 代表取締役社長   江澤　章 です。',
        '株式会社インプレスの従業員数は 123名（2023年4月1日現在） です。',
        '株式会社インプレスの住所は 〒103-0013 東京都中央区日本橋人形町2-26-5　NX人形町ビル5階 です。',
        '株式会社インプレスの連絡先は TEL：03-6914-8511  FAX：03-5643-6121 です。',
        'インプレスの主要なパッケージ商品は Excel運用サポートシステム「iFUSION」、連結決算システム「iCAS」、開示組替支援ツール「iFlap」 です。',
    ]
    
    chishikiVector = []
    for c in chishiki:
        chishikiVector.append({
            'text': c,
            'vector': createEmbedding(c)
        })
    return chishikiVector

# 入力テキストと知識ベクトルの類似度を計算し、上位3つを返す関数を定義します
def getRelevanceList(chishikiVector, user_text):
    def dot(a, b):
        return sum([a[i] * b[i] for i in range(len(a))])
    inputV = createEmbedding(user_text)
    similarities = [
        { 'text': i['text'],
        'similarity': dot(inputV, i['vector']) } 
        for i in chishikiVector
    ]
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:3]
    return [i['text'] for i in similarities]

# OpenAIのCompletion APIを使ってテキスト生成を行う関数を定義します
def createCompletion(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt
        )
        return response["choices"][0]["message"]["content"]   
    except Exception as e:
        print(e)
        raise e
        
# チャットボットの主要部分で、ユーザーからの入力テキストに対して回答を生成する関数を定義します
def chatbot(user_text):
    chishikiVector = createChishikiVector()
    relevanceList = getRelevanceList(chishikiVector, user_text)

    relevanceListText = '\n\n'.join(relevanceList)
    
    system_msg = f"""以下の制約条件に従って、株式会社インプレスのお問い合わせ窓口チャットボットとしてロールプレイをします。

---
# 制約条件:
- インプレスの情報を基に質問文に対する回答文を生成してください。
- 回答は見出し、箇条書き、表などを使って人間が読みやすく表現してください。

# インプレスの情報:
{relevanceListText}

# 回答文:
"""
    
    prompt = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text}
    ]

    # プロンプトを生成し、Completion APIを使用して回答を生成します
    completion = createCompletion(prompt)
    return completion

# Streamlit UIの定義
st.title("IMPRESS AI Assistant")
st.write("ChatGPT APIを使ったIMPチャットボットです。")

# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("質問を入力してください:")

if st.button("送信"):
    user_message = user_input.strip()
    if user_message:
        st.session_state["messages"].append({"role": "user", "content": user_message})
        bot_response = chatbot(user_message)
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})

if st.session_state["messages"]:
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.write(f"🙂: {message['content']}")
        else:
            st.write(f"🤖: {message['content']}")
