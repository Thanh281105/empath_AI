import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any
from tqdm import tqdm
from cskh_common import (
    add_common_args,
    resolve_api_settings,
    AsyncInferenceClient,
    normalize_messages,
    normalize_role,
    build_chatml_record,
    extract_json_payload,
    load_jsonl,
    load_jsonl_best_effort,
    write_jsonl
)

# --- PROMPTS & CONFIG FOR GROUP 1 (REJECTION) ---
SYSTEM_SYNTHETIC_PROMPT_G1 = """Bạn tạo dữ liệu fine-tune cho chatbot CSKH tiếng Việt.
Hãy tạo data "từ chối thấu cảm" (Group 1 - Rejection) chất lượng cao.

Yêu cầu cho mỗi mẫu:
- user: đổi trả/hoàn tiền trong tình huống RÕ RÀNG vi phạm quy định
- assistant: theo đúng công thức:
  1. Thấu cảm, xoa dịu
  2. Chỉ ra tình tiết vi phạm
  3. Từ chối dứt khoát, không đôi co
  4. Đưa giải pháp thay thế
- Văn phong tự nhiên, đời thường, đa dạng thái độ user: ngang, năn nỉ, dọa phốt, van xin, nóng nảy.
- Ưu tiên các case: rơi vỡ, vào nước, cháy mạch, quá hạn lâu, bóc seal bôi bẩn...
- Trả về JSON array hợp lệ.
"""

VIOLATION_SCENARIOS_G1 = [
    "bóc seal rồi mới đổi ý", "dùng gần một tuần rồi đòi đổi", "xài chán muốn trả lấy tiền lại",
    "quá hạn lâu mới contact", "làm rơi vỡ", "máy vào nước", "cắm sai điện áp",
    "lắp gãy ngàm", "giặt sai cách", "mất hộp phụ kiện", "tráo hàng cũ/fake"
]

# --- PROMPTS & CONFIG FOR GROUP 2 (FLEXIBLE/EXCEPTION) ---
SYSTEM_SYNTHETIC_PROMPT_G2 = """Bạn tạo dữ liệu fine-tune cho chatbot CSKH tiếng Việt.
Hãy tạo data "nhượng bộ thông minh / ngoại lệ đặc biệt" (Group 2 - Exceptions) chất lượng cao.

Yêu cầu cho mỗi mẫu:
- user: tình huống borderline/mập mờ hoặc khách VIP gặp khó khăn (quá hạn 1-2 ngày, bóc seal nhưng hàng brand new, lỗi không rõ do shipper hay khách).
- assistant: chiến thuật "Nhượng bộ giữ khách":
  1. Thấu cảm sâu sắc với tình cảnh của khách.
  2. Nhấn mạnh rằng theo quy định thì không được (để khách biết mình đang được ưu tiên).
  3. Đề xuất một "ngoại lệ duy nhất" hoặc "giải pháp linh động" (đổi mới, hoàn 80%, tặng voucher khủng).
  4. Tạo cảm giác khách hàng được trân trọng đặc biệt.
- Văn phong tự nhiên, chân thành, KHÔNG dùng văn mẫu rập khuôn.
- Trả về JSON array hợp lệ.
"""

SCENARIOS_G2 = [
    "Quá hạn đổi trả 1-2 ngày do khách đi công tác",
    "Bóc seal hộp nhưng sản phẩm chưa hề sử dụng, còn mới nguyên",
    "Lỗi nhỏ khách gặp phải nhưng khó chứng minh, shop quyết định tin tưởng khách",
    "Khách VIP mua nhiều lần, nay gặp sự cố do sơ suất cá nhân nhỏ",
    "Mất hộp giấy bên ngoài nhưng bộ phận chính vẫn còn tem mác",
    "Nhầm kích cỡ/màu sắc khi mua sale không được đổi, nhưng shop vẫn hỗ trợ đổi",
    "Hàng bị trầy xước nhẹ không rõ do ai, shop nhận trách nhiệm để khách vui"
]

USER_STYLES = [
    "năn nỉ ỉ ôi", "dọa bóc phốt", "cọc cằn", "ăn vạ cùn", "lịch sự nhưng ép linh động",
    "mặc cả xin ngoại lệ", "vin vào hoàn cảnh cá nhân", "đổ lỗi cho người nhà", "hối thúc"
]

def canonicalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered

def dedupe_batch_rows(rows):
    seen_pairs = set()
    deduped = []
    for row in rows:
        user_text = row["messages"][0]["content"]
        assistant_text = row["messages"][1]["content"]
        key = (canonicalize_text(user_text), canonicalize_text(assistant_text))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(row)
    return deduped

def parse_synthetic_payload(raw: str) -> list[dict[str, Any]]:
    return extract_json_payload(raw)

def build_row_from_item(item: dict[str, Any], batch_index: int, group: int) -> dict[str, Any]:
    msgs = normalize_messages(item)
    pair = {normalize_role(m["role"]): m["content"] for m in msgs}
    return build_chatml_record(
        user_text=pair["user"],
        assistant_text=pair["assistant"],
        metadata={
            "source": f"synthetic_group_{group}",
            "group": group,
            "scenario": item.get("scenario_tag", "gen_case"),
            "batch_index": batch_index
        }
    )

async def generate_batch(client, batch_size, batch_index, seed, group):
    system_prompt = SYSTEM_SYNTHETIC_PROMPT_G1 if group == 1 else SYSTEM_SYNTHETIC_PROMPT_G2
    scenarios = VIOLATION_SCENARIOS_G1 if group == 1 else SCENARIOS_G2
    
    collected = []
    seen_pairs = set()
    
    for attempt in range(3):
        remaining = batch_size - len(collected)
        if remaining <= 0: break
        
        scenario_list = ", ".join(random.sample(scenarios, k=min(len(scenarios), 4)))
        style_list = ", ".join(random.sample(USER_STYLES, k=min(len(USER_STYLES), 4)))
        
        prompt = f"""Tạo đúng {remaining} mẫu hội thoại. Group: {group}. Seed: {seed}.
        Đa dạng kịch bản từ: {scenario_list}
        Đa dạng thái độ từ: {style_list}
        
        Yêu cầu:
        - ChatML format: {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "scenario_tag": "..."}}
        - Trả về JSON array."""

        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            response_format={"type": "json_object"}
        )
        
        parsed_items = parse_synthetic_payload(raw)
        if not isinstance(parsed_items, list): parsed_items = [parsed_items]
        
        for item in parsed_items:
            if not isinstance(item, dict) or "messages" not in item: continue
            row = build_row_from_item(item, batch_index, group)
            u, a = row["messages"][0]["content"], row["messages"][1]["content"]
            key = (canonicalize_text(u), canonicalize_text(a))
            if key not in seen_pairs:
                seen_pairs.add(key)
                collected.append(row)
                
    return collected[:batch_size]

async def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation (G1 & G2)")
    add_common_args(parser)
    parser.add_argument("--group", type=int, choices=[1, 2], default=1, help="Group to generate (1: Rejection, 2: Exception)")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.output:
        args.output = Path(f"data/synthetic_group_{args.group}.jsonl")

    api_key, base_url, model, inf_type, v_project, v_location = resolve_api_settings(args)
    num_batches = (args.count + args.batch_size - 1) // args.batch_size
    
    completed_indices = set()
    if args.resume and args.output.exists():
        for row in load_jsonl_best_effort(args.output):
            if "batch_index" in row.get("metadata", {}):
                completed_indices.add(row["metadata"]["batch_index"])

    async with AsyncInferenceClient(
        model, api_key, base_url, inf_type, args.timeout_seconds, args.max_concurrency, 
        vertex_project=v_project, vertex_location=v_location
    ) as client:
        progress = tqdm(total=num_batches, desc=f"Generating G{args.group}", initial=len(completed_indices))
        lock = asyncio.Lock()
        
        async def worker(b_idx):
            try:
                start = b_idx * args.batch_size
                size = min(args.batch_size, args.count - start)
                res = await generate_batch(client, size, b_idx, args.seed + b_idx, args.group)
                async with lock:
                    with args.output.open("a", encoding="utf-8") as f:
                        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error batch {b_idx}: {e}")
            finally:
                progress.update(1)

        pending = [i for i in range(num_batches) if i not in completed_indices]
        await asyncio.gather(*(worker(i) for i in pending))

if __name__ == "__main__":
    asyncio.run(main())
