# Web 爬蟲與問答系統分析
這個程式是一個網頁爬蟲和基於 OpenAI 的問答系統，主要功能是爬取指定網站的內容，然後讓使用者可以向這些內容提出問題。整個流程分為以下幾個主要步驟：

1. 初始化與設定
導入必要的庫（requests, BeautifulSoup, OpenAI API 等）
設定 OpenAI API 金鑰
定義要爬取的網域（news.ltn.com.tw）
2. 網頁爬取功能
HyperlinkParser 類別：解析 HTML 並提取超連結
get_hyperlinks 函數：從指定 URL 獲取所有超連結
get_domain_hyperlinks 函數：過濾只保留同一網域內的連結
crawl 函數：使用廣度優先搜索爬取網站，並將內容保存為文字檔
3. 文字處理
remove_newlines 函數：清理文本中的換行符和多餘空格
從保存的文字檔中讀取內容並整理成資料框架
將標題和內容合併為單一文本
4. 文本分塊與標記化
使用 tiktoken 將文本轉換為標記
計算每個文本的標記數
split_into_many 函數：將長文本分割成較小的塊，以符合模型的輸入限制
5. 生成嵌入向量
使用 OpenAI 的 text-embedding-ada-002 模型為每個文本塊生成嵌入向量
保存嵌入向量到 CSV 檔案
6. 問答系統
distances_from_embeddings 函數：計算查詢嵌入與文本嵌入間的餘弦相似度
create_context 函數：找出與問題最相關的文本片段來構建上下文
answer_question 函數：基於構建的上下文使用 GPT-3.5-Turbo 回答問題
使用流程
程式會先爬取 news.ltn.com.tw 網站的內容
將爬取的內容處理並生成嵌入向量
使用者可以通過 answer_question 函數向系統提問
系統會找出與問題最相關的文本，並使用 GPT-3.5-Turbo 生成答案
這個程式實現了一個完整的 RAG (Retrieval-Augmented Generation) 系統，能夠基於特定網站的內容提供相關問答服務。