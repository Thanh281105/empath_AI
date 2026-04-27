import asyncio
import json
import os
import sys
import re
from pathlib import Path
from tqdm import asyncio as tq_asyncio

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from cskh_common import (
    resolve_api_settings,
    AsyncInferenceClient,
    load_jsonl,
    write_jsonl,
    extract_json_payload
)

# Configuration
OLD_DATA_PATH = Path(r"c:\Users\Admin\Desktop\vibe coding\data\old_data.jsonl")
AUDIT_REPORT_PATH = Path(r"c:\Users\Admin\Desktop\vibe coding\data\audit_report.json")
OUTPUT_PATH = Path(r"c:\Users\Admin\Desktop\vibe coding\data\old_data_dpo.jsonl")
CONCURRENCY = 20

SYSTEM_PROMPT = """Bạn là EmpathAI, chuyên gia huấn luyện CSKH. 
Nhiệm vụ của bạn là VIẾT LẠI câu phản hồi của nhân viên để trở nên chuyên nghiệp, thấu cảm và hiệu quả hơn.

QUY TẮC BẮT BUỘC:
1. KHÔNG dùng các câu lặp đi lặp lại như "Ôi trời, mình đọc xong cũng thấy nóng mặt thay bạn luôn".
2. KHÔNG dùng văn mẫu robot: "Chúng tôi rất tiếc", "Theo chính sách công ty", "Xin lỗi vì sự bất tiện".
3. THẤU CẢM thực sự: Diễn đạt sự hiểu biết về nỗi đau của khách bằng nhiều cách khác nhau (ví dụ: "Mình hiểu cảm giác hụt hẫng khi...", "Thật khó chấp nhận khi...", "Mình rất lấy làm tiếc vì trải nghiệm này...").
4. ĐỀ XUẤT bồi thường cụ thể (voucher, tiền mặt, quà tặng) dựa trên bối cảnh.
5. KẾT THÚC bằng một câu hỏi mở để giữ liên lạc.
6. GIỮ NGUYÊN các con số bồi thường nếu có trong câu gốc, trừ khi nó quá vô lý.

FORMAT ĐẦU RA (Chỉ trả về JSON):
{"chosen": "nội dung câu phản hồi mới"}"""

ROBOTIC_INTRO = [
    "Chào bạn, chúng tôi đã nhận được thông tin khiếu nại của bạn về đơn hàng.",
    "Hệ thống ghi nhận khiếu nại của bạn. Theo chính sách công ty, chúng tôi sẽ phản hồi sau.",
    "Xin lỗi vì sự bất tiện này. Chúng tôi sẽ chuyển thông tin cho bộ phận liên quan xử lý.",
    "Chào bạn, Shop đã tiếp nhận yêu cầu. Vui lòng chờ trong giây lát."
]

async def process_record(client, record, index, is_bad):
    messages = record.get("messages", [])
    if not messages or len(messages) < 2:
        return None

    prompt_messages = messages[:-1] # System + User
    original_assistant = messages[-1]["content"]
    
    # Context for Gemini
    user_query = messages[-2]["content"] if len(messages) > 1 else ""
    
    if is_bad:
        # Need Gemini to fix it
        user_input = f"Câu phản hồi gốc (TỆ/LỖI): {original_assistant}\n\nBối cảnh khách nói: {user_query}\n\nHãy viết lại câu 'chosen' tuyệt vời nhất."
        
        try:
            response = await client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7
            )
            data = extract_json_payload(response)
            chosen_content = data.get("chosen", original_assistant)
            rejected_content = original_assistant
        except Exception as e:
            print(f"Error fixing index {index}: {e}")
            return None
    else:
        # Already good, make a robotic rejected version to save cost
        import random
        chosen_content = original_assistant
        # Create a robotic version
        parts = original_assistant.split('.', 1)
        if len(parts) > 1:
            body = parts[1].strip()
        else:
            body = original_assistant
        
        rejected_content = random.choice(ROBOTIC_INTRO) + " " + body

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": chosen_content}],
        "rejected": [{"role": "assistant", "content": rejected_content}]
    }

async def main():
    import argparse
    from cskh_common import add_common_args
    
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    
    api_key, base_url, model, inf_type, v_proj, v_loc = resolve_api_settings(args)
    client = AsyncInferenceClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        inference_type=inf_type,
        timeout_seconds=args.timeout_seconds,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.requests_per_minute,
        vertex_project=v_proj,
        vertex_location=v_loc
    )
    
    print(f"Loading data...")
    records = load_jsonl(OLD_DATA_PATH)
    
    with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
        audit_report = json.load(f)
    
    bad_indices = {item["index"] for item in audit_report}
    
    print(f"Starting DPO conversion. Focusing ONLY on {len(bad_indices)} bad rows. Skipping 709 good rows.")
    
    async with client:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        tasks = []
        
        async def sem_process(idx, record):
            async with semaphore:
                # is_bad is always True here because of filtering below
                return await process_record(client, record, idx, True)

        # Only create tasks for bad indices
        for i, record in enumerate(records):
            if i in bad_indices:
                tasks.append(sem_process(i, record))
        
        results = []
        # Support recovery: load existing results if any
        if OUTPUT_PATH.exists():
             results = load_jsonl(OUTPUT_PATH)
             completed_prompts = {json.dumps(r["prompt"], sort_keys=True) for r in results}
             print(f"Skipping {len(completed_prompts)} already processed records.")
             # Filter tasks
             new_tasks = []
             for i, record in enumerate(records):
                 p = record.get("messages", [])[:-1]
                 if json.dumps(p, sort_keys=True) not in completed_prompts:
                     new_tasks.append(tasks[i])
             tasks = new_tasks

        if not tasks:
            print("Everything already processed.")
            return

        for f in tq_asyncio.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            res = await f
            if res:
                results.append(res)
                if len(results) % 50 == 0:
                    write_jsonl(OUTPUT_PATH, results)

        write_jsonl(OUTPUT_PATH, results)
        print(f"Successfully converted {len(results)} records to DPO format at {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
