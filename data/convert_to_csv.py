import pandas as pd
import numpy as np
import os

def join_list(val):
    if isinstance(val, (list, np.ndarray)):
        return "; ".join([str(x) for x in val if x is not None])
    return val

def convert():
    base_dir = "/Users/yoichiro/Library/CloudStorage/GoogleDrive-y.kanno@sakenote.net/マイドライブ/サケノテ/python/論文DB"
    output_dir = os.path.join(base_dir, "JBSJ_P/data")
    
    # 1. paper_db_final.parquet -> keywords_summary5.csv & summaries.csv
    paper_db_path = os.path.join(base_dir, "8_reHP/4.marge/paper_db_final.parquet")
    if os.path.exists(paper_db_path):
        df = pd.read_parquet(paper_db_path)
        
        # keywords_summary5.csv
        kw_mapping = {
            'No.': 'No.',
            'rel_path': '相対PASS',
            '発行年': '発行年',
            '巻数': '巻数',
            '号数': '号数',
            '開始ページ': '開始ページ',
            '終了ページ': '終了ページ',
            '論文タイトル': '論文タイトル',
            '著者': '著者',
            'file_name': 'file_name',
            'HPリンク先': 'HPリンク先',
            'PDFリンク先': 'PDFリンク先',
            'llm_keywords': 'llm_keywords',
            'primary_keywords': 'primary_keywords',
            'secondary_keywords': 'secondary_keywords',
            'targets_all': '対象物_all',
            'targets_top3': '対象物_top3',
            'targets_rationale': '対象物_根拠',
            'research_type_all': '研究タイプ_all',
            'research_type_top3': '研究タイプ_top3',
            'research_type_rationale': '研究タイプ_根拠'
        }
        
        df_kw = df[list(kw_mapping.keys())].rename(columns=kw_mapping)
        
        # リスト型の列をセミコロンで結合
        list_cols = ['llm_keywords', 'primary_keywords', 'secondary_keywords', '対象物_all', '対象物_top3', '研究タイプ_all', '研究タイプ_top3']
        for col in list_cols:
            if col in df_kw.columns:
                df_kw[col] = df_kw[col].apply(join_list)
        
        # 不足している列を追加
        df_kw['file_path'] = ""
        df_kw['num_pages'] = ""
        df_kw['featured_keywords'] = ""
        
        # 列の順序を調整
        kw_cols_order = [
            'No.', '相対PASS', '発行年', '巻数', '号数', '開始ページ', '終了ページ', '論文タイトル', '著者', 'file_name', 
            'HPリンク先', 'PDFリンク先', 'file_path', 'num_pages', 'llm_keywords', 'primary_keywords', 
            'secondary_keywords', 'featured_keywords', '対象物_all', '対象物_top3', '対象物_根拠', 
            '研究タイプ_all', '研究タイプ_top3', '研究タイプ_根拠'
        ]
        df_kw = df_kw[kw_cols_order]
        df_kw.to_csv(os.path.join(output_dir, "keywords_summary5.csv"), index=False, encoding='utf-8-sig')
        print(f"Created: keywords_summary5.csv ({len(df_kw)} rows)")
        
        # summaries.csv
        df_sum = df[['file_name', 'rel_path', 'catchphrase', 'key_result']].copy()
        df_sum.columns = ['file_name', 'rel_path', 'summary', 'summary_source_text']
        df_sum.to_csv(os.path.join(output_dir, "summaries.csv"), index=False, encoding='utf-8-sig')
        print(f"Created: summaries.csv ({len(df_sum)} rows)")
    else:
        print(f"Error: {paper_db_path} not found.")

    # 2. authors_readings_updated.parquet -> authors_readings.csv
    author_path = os.path.join(base_dir, "8_reHP/5.author/authors_readings_updated.parquet")
    if os.path.exists(author_path):
        df_author = pd.read_parquet(author_path)
        df_author.to_csv(os.path.join(output_dir, "authors_readings.csv"), index=False, encoding='utf-8-sig')
        print(f"Created: authors_readings.csv ({len(df_author)} rows)")
    else:
        print(f"Error: {author_path} not found.")

if __name__ == "__main__":
    convert()
