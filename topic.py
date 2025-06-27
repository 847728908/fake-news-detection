from pathlib import Path
import re
import nltk
from gensim import corpora, models
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import ollama  # 确保已安装ollama并本地有模型
import matplotlib

# 确保nltk资源已下载
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

topic_analyses = []

def get_post_text_batches(posts_file, batch_size=10):
    post_texts = []
    with open(posts_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过标题行
        post_text_list = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                post_text_list.append(parts[1])
        # 按batch_size分组拼接
        for i in range(0, len(post_text_list), batch_size):
            batch = post_text_list[i:i+batch_size]
            batch_text = '\n'.join(batch)
            post_texts.append(batch_text)
    return post_texts

def preprocess_text(text, stop_words, lemmatizer):
    # 1. 清洗：去除非字母字符，转小写
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    # 2. 分词
    words = text.split()
    # 3. 去停用词和短词
    words = [w for w in words if w not in stop_words and len(w) > 2]
    # 4. 词形还原
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

if __name__ == "__main__":
    posts_file = r"C:\Users\84772\Desktop\try\twitter_dataset\devset\posts.txt"
    output_file = r"C:\Users\84772\Desktop\try\concatenate10.txt"
    processed_file = r"C:\Users\84772\Desktop\try\concatenate10_processed.txt"

    if not Path(posts_file).exists() or Path(posts_file).stat().st_size == 0:
        print(f"{posts_file} 文件不存在或为空，请检查后再运行。")
    else:
        # 检查processed_file是否存在且非空
        if Path(processed_file).exists() and Path(processed_file).stat().st_size > 0:
            print(f"检测到已存在的预处理文件 {processed_file}，直接读取。")
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_batches = [line.strip() for line in f if line.strip()]
        else:
            batches = get_post_text_batches(posts_file, batch_size=10)
            # 将每组写入一行
            with open(output_file, 'w', encoding='utf-8') as f:
                for batch in batches:
                    f.write(batch.replace('\n', ' ') + '\n')  # 每组10条合成一行，原有换行替换为空格
            print(f"已保存到 {output_file}")

            # 预处理部分
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            processed_batches = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_line = preprocess_text(line, stop_words, lemmatizer)
                    processed_batches.append(processed_line)
            with open(processed_file, 'w', encoding='utf-8') as f:
                for line in processed_batches:
                    f.write(line + '\n')
            print(f"预处理后已保存到 {processed_file}")

        # ====== 新增：LDA主题模型部分 ======
        # 1. 构建词典
        # 先将每行分词
        texts = [line.split() for line in processed_batches]
        dictionary = corpora.Dictionary(texts)
        print(f"词典大小: {len(dictionary)}")

        # 2. 生成语料库（词袋模型）
        corpus = [dictionary.doc2bow(text) for text in texts]

        # 3. 训练LDA模型
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=5,  # 主题数可根据需要调整
            random_state=42,
            passes=10
        )

        # 打印每个主题的关键词
        for idx, topic in lda_model.print_topics(num_words=10):
            print(f"主题{idx}: {topic}")

        # 可选：保存模型和字典
        dictionary.save(r"C:\Users\84772\Desktop\try\lda_dictionary.dict")
        lda_model.save(r"C:\Users\84772\Desktop\try\lda_model.model")
        print("LDA模型和词典已保存。")

        # 1. pyLDAvis交互图
        pyldavis_output = r"C:\\Users\\84772\\Desktop\\try\\lda_pyldavis.html"
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, pyldavis_output)
        print(f"pyLDAvis交互图已保存到 {pyldavis_output}")

        # 保存后自动插入居中div
        html_path = 'lda_pyldavis.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # 用正则找到 id="ldavis_elxxxx" 的div，并加上外层div
        import re
        pattern = r'(<div id="ldavis_el[0-9]+".*?</div>)'
        replacement = r'<div style="width: 100%; display: flex; justify-content: center;">\n\1\n</div>'
        html_new = re.sub(pattern, replacement, html, count=1, flags=re.DOTALL)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_new)

        # 2. 词云图
        wordcloud_dir = r"C:\\Users\\84772\\Desktop\\try\\lda_wordclouds"
        Path(wordcloud_dir).mkdir(exist_ok=True)
        for t in range(lda_model.num_topics):
            plt.figure(figsize=(12, 6), dpi=200)  # 提高分辨率
            topic_words = dict(lda_model.show_topic(t, topn=30))
            wc = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(topic_words)  # 提高像素
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {t} Wordcloud', fontsize=20)  # 英文标题，字号更大
            wc_path = f'{wordcloud_dir}/topic_{t}_wordcloud.png'
            plt.savefig(wc_path, bbox_inches='tight')
            plt.close()
            print(f'主题{t}词云已保存到 {wc_path}')

        # 3. 热力图
        doc_topic_matrix = np.array([lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus])
        # 转为二维概率矩阵
        doc_topic_probs = np.array([[prob for _, prob in doc] for doc in doc_topic_matrix])
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

        plt.figure(figsize=(10, 6))
        sns.heatmap(doc_topic_probs, cmap='YlGnBu', cbar=True)
        plt.xlabel('Topic', fontsize=14)
        plt.ylabel('Document', fontsize=14)
        plt.title('Document-Topic Probability Heatmap', fontsize=16)
        heatmap_path = r"C:\\Users\\84772\\Desktop\\try\\lda_doc_topic_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f'热力图已保存到 {heatmap_path}')

        # 4. 结合大模型分析主题内容
        def analyze_topic_with_llm(topic_keywords, topic_id):
            prompt = f"""你是一名专业的主题分析专家。请根据以下关键词，分析该主题的主要内容、潜在含义和可能涉及的社会话题，并给出简要总结。
            
主题编号：{topic_id}
关键词：{', '.join(topic_keywords)}

请用简洁的中文进行分析和总结。"""
            try:
                response = ollama.chat(
                    model='deepseek-r1:7b',  # 可根据本地模型实际名称调整
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
                analysis = response['message']['content']
                print(f"主题{topic_id}分析：\n{analysis}\n")
                topic_analyses.append(f"主题{topic_id}分析：\n{analysis}\n")
            except Exception as e:
                error_msg = f"主题{topic_id}大模型分析出错: {str(e)}\n"
                print(error_msg)
                topic_analyses.append(error_msg)

        print("\n===== 主题内容分析（大模型自动分析） =====\n")
        for t in range(lda_model.num_topics):
            topic_words = [w for w, _ in lda_model.show_topic(t, topn=10)]
            analyze_topic_with_llm(topic_words, t)

        analysis_output = r"C:\Users\84772\Desktop\try\lda_topic_analysis.txt"
        with open(analysis_output, 'w', encoding='utf-8') as f:
            for analysis in topic_analyses:
                f.write(analysis + '\n')
        print(f"主题分析结果已保存到 {analysis_output}")
