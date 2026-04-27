import json
import os
from pathlib import Path

def audit_data(file_path):
    issues = []
    repetitive_phrase = "nóng mặt thay bạn"
    forbidden_phrases = ["Chúng tôi rất tiếc", "Theo chính sách công ty", "Xin lỗi vì sự bất tiện"]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages:
                    issues.append({"index": i, "reason": "No messages found"})
                    continue
                
                assistant_msg = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
                
                line_issues = []
                if repetitive_phrase.lower() in assistant_msg.lower():
                    line_issues.append("Repetitive empathetic phrase")
                
                for fp in forbidden_phrases:
                    if fp.lower() in assistant_msg.lower():
                        line_issues.append(f"Contains forbidden phrase: '{fp}'")
                
                if len(assistant_msg) < 50:
                    line_issues.append("Response too short")
                    
                if line_issues:
                    issues.append({"index": i, "reasons": line_issues, "content": assistant_msg[:100] + "..."})
                    
            except json.JSONDecodeError:
                issues.append({"index": i, "reason": "JSON Decode Error"})

    return issues

if __name__ == "__main__":
    old_data_path = r"c:\Users\Admin\Desktop\vibe coding\data\old_data.jsonl"
    report = audit_data(old_data_path)
    
    with open(r"c:\Users\Admin\Desktop\vibe coding\data\audit_report.json", "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    
    print(f"Audit complete. Found {len(report)} lines with issues out of 3408.")
    print("Report saved to data/audit_report.json")
