import os
import re
import pandas as pd
from email.parser import Parser
from email.policy import default

OUTPUT_CSV = "spam_assassin.csv"
HAM_DIR = "20021010_easy_ham/easy_ham"
SPAM_DIR = "spam"

def parse_email_content(content):
    try:
        parser = Parser(policy=default)
        email = parser.parsestr(content)
        subject = email['subject'] or ''
        body = email.get_body(preferencelist=('plain', 'html'))
        if body:
            text = body.get_content()
            text = re.sub(r'\s+', ' ', text).strip()
            return subject + ' ' + text
        return subject
    except:
        return content[:500] if content else ""

def load_emails_from_dir(directory, label):
    emails = []
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return emails
    
    print(f"从 {directory} 加载邮件 (label={label})...")
    files = os.listdir(directory)
    
    for i, filename in enumerate(files):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                    content = f.read()
                    if 'From ' in content and '@' in content:
                        text = parse_email_content(content)
                        if text and len(text) > 20:
                            emails.append({'text': text, 'target': label})
            except:
                continue
        
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(files)} 个文件, 获取 {len(emails)} 封邮件")
    
    print(f"  共获取 {len(emails)} 封邮件")
    return emails

def create_dataset():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"数据集已存在: {OUTPUT_CSV}")
        print(f"  总数: {len(df)}, 正常: {len(df[df['target']==0])}, 垃圾: {len(df[df['target']==1])}")
        return df
    
    print("开始从本地文件夹加载邮件数据...")
    
    all_emails = []
    all_emails.extend(load_emails_from_dir(HAM_DIR, 0))
    all_emails.extend(load_emails_from_dir(SPAM_DIR, 1))
    
    if not all_emails:
        print("错误: 未能获取任何邮件")
        return None
    
    df = pd.DataFrame(all_emails)
    df = df.drop_duplicates(subset=['text'])
    df = df[df['text'].str.len() > 20]
    
    print(f"\n数据集统计:")
    print(f"  总邮件数: {len(df)}")
    print(f"  正常邮件: {len(df[df['target']==0])}")
    print(f"  垃圾邮件: {len(df[df['target']==1])}")
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n数据集已保存到 {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    create_dataset()