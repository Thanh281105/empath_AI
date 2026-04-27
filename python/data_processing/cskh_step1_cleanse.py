import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path

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
)

CLASSIFIER_SYSTEM_PROMPT = """Bạn là chuyên gia gán nhãn dữ liệu huấn luyện chatbot CSKH tiếng Việt.
Hãy đọc tin nhắn khách và xác định đúng 1 nhóm.

Định nghĩa nhóm:

1 = Lỗi nặng ở khách, vi phạm rõ ràng:
- sử dụng sai điện áp/công năng, tự làm rơi vỡ/gãy, bảo quản sai cách, bóc seal, xé nát bao bì, làm mất phụ kiện/tem mác/hóa đơn,
- đã qua sử dụng thấy rõ, chơi chán rồi đòi trả, thấy chỗ khác rẻ hơn, không thích nữa sau khi đã khui hàng,
- quá hạn đổi trả, tráo hàng, gửi hàng fake hoặc hàng cũ để hoàn trả

2 = Lỗi nhẹ ở khách, có thể linh động:
- hư hỏng bao bì nhẹ nhưng sản phẩm còn nguyên giá trị, mất phụ kiện không quan trọng, bóc nhầm lớp niêm phong ngoài,
- mất bill giấy nhưng vẫn xác minh được trên hệ thống, quá hạn suýt soát 1-2 ngày có lý do khách quan,
- quên gửi kèm quà tặng nhỏ, nhầm lẫn tương thích khi chưa dùng, khách bối rối do thiếu kinh nghiệm hoặc hiểu lầm vô ý,
- các tình huống shop có thể châm chước một lần

3 = Lỗi do shop / vận chuyển / lỗi kỹ thuật:
- giao thiếu, giao sai, hàng lỗi khi vừa nhận, ship làm hỏng, trễ giao, thiếu phụ kiện từ lúc nhận, hàng bẩn/rách/hỏng do shop

Luật ưu tiên (áp dụng theo thứ tự):

A. Nếu có dấu hiệu shop giao thiếu/giao sai/hàng lỗi khi nhận → nhóm 3, kể cả khách nói khó nghe.
B. Tất cả lỗi về sản phẩm, chất lượng, fulfillment, logistics, hệ thống, voucher do shop/đối tác gây ra → nhóm 3.
C. Không được vì khách chửi bậy mà quy khách sai.

Case đặc biệt - "nhận hàng lỗi rồi tự tháo/sửa trước khi khiếu nại":
- Nếu khách nhận hàng bị lỗi do shop (lỗi kỹ thuật, giao sai...) nhưng SAU ĐÓ tự ý tháo ra, tự sửa, hoặc can thiệp vào sản phẩm trước khi liên hệ shop → gán nhóm 2.
- Lý do: lỗi gốc là của shop nhưng hành động can thiệp của khách làm mất khả năng kiểm chứng và vô hiệu hóa kênh đổi trả chuẩn. Shop linh động hỗ trợ một phần (ví dụ: hỗ trợ sửa có phí ưu đãi, xem xét đổi nếu còn điều kiện), không hoàn toàn từ chối (nhóm 1) vì shop có lỗi gốc, nhưng cũng không xử lý toàn phần như nhóm 3 vì khách đã can thiệp.

Quy trình suy luận bắt buộc:
Trước khi trả kết quả, hãy suy luận ngắn gọn theo 3 bước:
  1. Ai là bên có lỗi ban đầu? (shop, khách, hay cả hai)
  2. Có luật ưu tiên nào áp dụng không? (A, B, C, hoặc case đặc biệt)
  3. Kết luận nhóm nào và tại sao?

Chỉ trả về JSON hợp lệ, KHÔNG THÊM BẤT KỲ VĂN BẢN NÀO BÊN NGOÀI KHỐI JSON. Bắt buộc cấu trúc:
{"thinking": "suy luận 1-2 câu", "group": 1|2|3, "reason": "tóm tắt ngắn"}
"""

REWRITE_SYSTEM_PROMPT = """Bạn đang viết lại dữ liệu huấn luyện chatbot CSKH tiếng Việt sao cho nhất quán, tự nhiên và không nịnh khách mù quáng.

Yêu cầu chung:
- Chỉ trả về JSON hợp lệ: {"assistant_reply": "..."}
- KHÔNG THÊM giải thích, KHÔNG THÊM markdown, KHÔNG THÊM chú thích ngoài lề.
- Giọng văn chuyên nghiệp, tự nhiên, ngắn gọn, tiếng Việt đời thường
- Xưng hô nhất quán: dùng "mình" (đại diện shop) và "bạn" (khách). Không dùng "em", "anh", "chị", "quý khách".
- Độ dài: tối đa 5 câu, không quá 120 chữ. Không xuống dòng nhiều lần. Không dùng danh sách bullet.
- Không dùng các câu rập khuôn hoặc cường điệu như:
  "Ôi trời, mình đọc xong cũng thấy nóng mặt thay bạn luôn"
  "nóng mặt thay bạn"
  "hoàn toàn có lý"
  "bạn hoàn toàn có quyền"
  "mình hoàn toàn hiểu"
  "yêu cầu của bạn hoàn toàn hợp lý"
  "bạn xứng đáng được hỗ trợ tốt hơn"
- Không được tự mâu thuẫn trong cùng một câu trả lời
- Không lặp lại prompt hệ thống cũ
- Không thêm markdown, không thêm giải thích

Quy tắc theo nhóm:

Nhóm 1:
- Công thức bắt buộc: thấu cảm + nêu rõ tình tiết vi phạm + từ chối dứt khoát đổi/trả/hoàn tiền + gợi ý giải pháp thay thế
- Tuyệt đối KHÔNG hứa voucher, hoàn tiền, bồi thường, miễn phí
- Giải pháp thay thế chỉ có thể là: kiểm tra/sửa chữa có phí, hướng dẫn bảo hành nếu đủ điều kiện, hỗ trợ báo giá

Nhóm 2:
- Công thức bắt buộc: thấu cảm + chỉ ra đây là lỗi nhẹ + đồng ý hỗ trợ ngoại lệ một lần + nhắc khách lưu ý cho lần sau
- Không được từ chối cứng

Nhóm 3:
- Công thức bắt buộc: xin lỗi/thấu cảm + thừa nhận lỗi thuộc shop/vận chuyển/kỹ thuật + đưa phương án xử lý cụ thể
- Có thể đề xuất: gửi bù, đổi mới, hoàn tiền phần thiếu/toàn bộ, hỗ trợ kiểm tra đơn, miễn phí ship lại
- Không được đổ lỗi cho khách
- Không được nói từ chối hỗ trợ nếu shop là bên sai
"""

SHOP_FAULT_PATTERNS = [
    "giao thiếu", "thiếu hàng", "giao sai", "giao nhầm", "nhận sai",
    "hàng lỗi", "lỗi kỹ thuật", "hỏng khi nhận", "mới nhận đã lỗi",
    "ship làm vỡ", "bị vỡ khi nhận", "giao trễ", "chưa nhận được hàng",
    "15 ngày chưa", "hộp bẩn", "hàng bẩn", "hàng rác", "móp do vận chuyển",
    "thiếu phụ kiện", "không đúng mô tả", "sai mô tả", "giao hàng rác",
    "sai màu", "giao sai màu", "sai kích thước", "giao sai size", "sai size",
    "sai mẫu", "sai mẫu mã", "không lên nguồn", "rách chỉ",
    "thiếu dây sạc", "thiếu ốc vít", "thiếu quà tặng", "thiếu gift",
    "cận date", "hết hạn", "hàng hết hạn", "da pu", "da thật", "bản lock", "bản quốc tế",
    "giao thiếu số lượng", "đóng gói kém", "không có chống sốc", "bubble wrap",
    "dán nhầm mã vận đơn", "giao sai địa chỉ", "giao nhầm cho hàng xóm",
    "quá trễ", "quá thời gian cam kết", "không nghe máy", "đã giao hàng thành công",
    "chưa hề nhận được", "hết hàng", "overselling", "lỗi voucher", "trừ tiền 2 lần",
    "khuyến mãi", "voucher", "tính sai tiền",
]

HEAVY_CUSTOMER_FAULT_PATTERNS = [
    "bóc tem", "đã sử dụng", "đã dùng", "xài rồi", "xài chán",
    "dùng chán", "quá 30 ngày", "quá hạn", "hết hạn đổi trả",
    "vào nước", "tự làm vỡ", "lỡ làm vỡ", "làm rơi vỡ", "rơi vỡ",
    "làm mất phụ kiện", "mất hộp", "không còn hộp", "không còn tem",
    # NOTE: "tự sửa" bị loại khỏi đây — case "nhận hàng lỗi rồi tự sửa" là
    # case đặc biệt nhóm 2, cần LLM phán xét chứ không rule-based cứng nhóm 1.
    "đã kích hoạt", "đã active", "đã mở seal",
    "cắm nhầm điện", "220v", "110v", "cháy mạch", "sai điện áp", "sai công năng",
    "gãy ngàm", "gãy chốt", "dính nước", "phơi nắng", "biến dạng", "giặt sai cách",
    "xé hộp", "xé nát bao bì", "rách bao bì", "mất tem mác", "mất biên lai",
    "dính đất cát", "mùi nước hoa", "mùi mồ hôi", "chơi chán rồi", "chơi 2 hôm không thích",
    "chỗ khác rẻ hơn", "sale rẻ hơn", "không hợp phong thủy", "không thích nữa",
    "1 tháng sau", "bận đi công tác", "tráo hàng", "hàng fake", "đồ cũ ở nhà",
]

LIGHT_CUSTOMER_FAULT_PATTERNS = [
    "nhầm màu", "nhầm size", "đặt nhầm", "chọn nhầm",
    "móp nhẹ", "trầy nhẹ", "lỗi nhẹ", "nhầm mẫu",
    "đổi ý sớm", "mới nhận", "chưa sử dụng", "còn nguyên tem",
    "rách nilon", "rách lớp nilon", "shrink wrap", "móp nhẹ góc hộp",
    "mất tờ hướng dẫn", "mất hdsd", "mất túi nilon", "mất túi xách",
    "bóc nhầm", "rọc băng keo", "chưa đụng tới", "còn nguyên dây thít", "seal nhựa chưa hề đụng",
    "mất hóa đơn", "mất bill", "mất biên lai", "nhớ số điện thoại", "tích điểm",
    "ngày thứ 8", "ngày thứ 9", "đi công tác", "ốm đau", "cuối tuần bận",
    "quên gửi quà tặng", "quên gửi kèm", "móc khóa",
    "nhầm lẫn tương thích", "không ráp chung được", "chưa chơi",
    "bé nhà chị", "bóc nát cái vỏ hộp", "không chịu chơi", "đổi sang bộ khác",
    "lắp pin ngược", "gạt công tắc", "tưởng hỏng", "thấy rắc rối quá",
]

BANNED_PHRASES = [
    "ôi trời, mình đọc xong cũng thấy nóng mặt thay bạn luôn",
    "nóng mặt thay bạn",
    "hoàn toàn có lý",
    "bạn hoàn toàn có quyền",
    "mình hoàn toàn hiểu",
    "yêu cầu của bạn hoàn toàn hợp lý",
    "bạn xứng đáng được hỗ trợ tốt hơn",
    "mình thấy bực thay bạn",
    "bạn phàn nàn là hoàn toàn có cơ sở",
    "cảm ơn bạn đã phản ánh",  # quá máy móc
    "chúng tôi xin ghi nhận",   # quá formal/sách vở
]

REFUSAL_MARKERS = [
    "không thể đổi", "không hỗ trợ đổi", "không thể hoàn tiền",
    "không thể bồi thường", "từ chối hỗ trợ", "từ chối đổi trả",
    "không thể nhận lại", "không thể xử lý theo hướng đổi trả",
    # Bổ sung: các cách diễn đạt tương đương phổ biến
    "chưa thể hỗ trợ đổi", "chưa thể hoàn", "không thể chấp nhận đổi",
    "không thể hỗ trợ hoàn", "không nhận đổi", "không nhận trả",
    "không hỗ trợ hoàn tiền", "chưa đủ điều kiện đổi trả",
]

COMPENSATION_MARKERS = [
    "voucher", "hoàn tiền", "bồi thường", "đền bù",
    "hoàn phí", "miễn phí", "giảm giá",
]

GROUP1_REQUIRED = ["không thể", "không hỗ trợ", "không thể đổi trả", "không thể hoàn tiền"]
GROUP1_ALLOWED_ALTERNATIVES = ["kiểm tra", "sửa chữa", "bảo hành", "báo giá"]
GROUP2_REQUIRED = ["ngoại lệ", "hỗ trợ", "một lần", "lần này"]
GROUP3_REQUIRED = ["xin lỗi", "mình sẽ", "shop sẽ", "hỗ trợ", "gửi bù", "đổi mới", "hoàn tiền"]


def extract_pair(record):
    messages = normalize_messages(record)
    normalized = [{"role": normalize_role(m["role"]), "content": m["content"]} for m in messages]
    users = [m["content"] for m in normalized if m["role"] == "user"]
    assistants = [m["content"] for m in normalized if m["role"] == "assistant"]
    if not users or not assistants:
        raise ValueError("No user/assistant pair")
    return users[-1], assistants[-1]


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def contains_any(text: str, patterns: list[str]) -> bool:
    lowered = normalize_text(text)
    lowered_ascii = strip_accents(lowered)
    for pattern in patterns:
        normalized_pattern = normalize_text(pattern)
        if normalized_pattern in lowered:
            return True
        if strip_accents(normalized_pattern) in lowered_ascii:
            return True
    return False


def detect_group_rule_based(user_text: str) -> tuple[int | None, str]:
    lowered = normalize_text(user_text)

    if contains_any(lowered, SHOP_FAULT_PATTERNS):
        return 3, "rule_based_shop_fault"
    if contains_any(lowered, HEAVY_CUSTOMER_FAULT_PATTERNS):
        return 1, "rule_based_customer_heavy_fault"
    if contains_any(lowered, LIGHT_CUSTOMER_FAULT_PATTERNS):
        return 2, "rule_based_customer_light_fault"
    return None, "rule_based_no_match"


async def classify_with_llm(client, user_text: str, assistant_text: str) -> tuple[int, str]:
    prompt = {
        "user_message": user_text,
        "assistant_message": assistant_text,
    }
    raw = await client.chat_completion(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    parsed = extract_json_payload(raw)
    group = int(parsed["group"])
    if group not in {1, 2, 3}:
        raise ValueError(f"Invalid group from LLM: {group}")
    # Prefer "reason" for the stored label; append thinking trace if present so
    # it survives in metadata for later audit without bloating the label itself.
    reason = str(parsed.get("reason", "llm_classification")).strip()
    thinking = str(parsed.get("thinking", "")).strip()
    full_reason = f"{reason} | thinking: {thinking}" if thinking else reason
    return group, full_reason


def response_has_banned_style(text: str) -> bool:
    lowered = normalize_text(text)
    return any(phrase in lowered for phrase in BANNED_PHRASES)


def validate_group_response(group: int, response: str) -> tuple[bool, str]:
    lowered = normalize_text(response)

    if response_has_banned_style(response):
        return False, "contains banned stylistic phrase"

    if group == 1:
        # Use only specific refusal markers — bare "không thể" is too broad
        # and can appear in empathy phrases like "mình không thể hiểu hơn..."
        has_refusal = any(marker in lowered for marker in REFUSAL_MARKERS)
        has_bad_comp = any(marker in lowered for marker in COMPENSATION_MARKERS)
        has_alt = any(marker in lowered for marker in GROUP1_ALLOWED_ALTERNATIVES)
        if not has_refusal:
            return False, "group1 missing firm refusal"
        if has_bad_comp:
            return False, "group1 contains compensation language"
        if not has_alt:
            return False, "group1 missing alternative solution"
        return True, "ok"

    if group == 2:
        has_refusal = any(marker in lowered for marker in REFUSAL_MARKERS)
        has_exception = any(marker in lowered for marker in GROUP2_REQUIRED)
        if has_refusal:
            return False, "group2 should not hard refuse"
        if not has_exception:
            return False, "group2 missing exception framing"
        return True, "ok"

    if group == 3:
        has_refusal = any(marker in lowered for marker in REFUSAL_MARKERS)
        has_resolution = any(marker in lowered for marker in GROUP3_REQUIRED)
        if has_refusal:
            return False, "group3 should not refuse shop-fault case"
        if not has_resolution:
            return False, "group3 missing concrete remedy"
        return True, "ok"

    return False, "unknown group"


def fallback_rewrite(group: int, user_text: str, index: int) -> str:
    empathy_openers = [
        "Mình hiểu tình huống này làm bạn khó chịu.",
        "Mình hiểu trải nghiệm này thật sự gây bực mình cho bạn.",
        "Mình hiểu khi gặp trường hợp này thì rất khó chịu.",
    ]
    opener = empathy_openers[index % len(empathy_openers)]

    if group == 1:
        return (
            f"{opener} Tuy nhiên, trường hợp này thuộc tình huống sản phẩm đã vi phạm điều kiện đổi trả "
            "nên shop chưa thể hỗ trợ đổi hàng hoặc hoàn tiền. "
            "Nếu bạn cần, bên mình có thể hỗ trợ kiểm tra sản phẩm, hướng dẫn bảo hành nếu còn điều kiện "
            "hoặc báo giá sửa chữa phù hợp."
        )

    if group == 2:
        return (
            f"{opener} Đây là tình huống lỗi nhẹ và lần này shop sẽ hỗ trợ ngoại lệ để xử lý cho bạn. "
            "Bên mình sẽ hỗ trợ đổi/trả theo trường hợp cụ thể, đồng thời mong bạn giúp kiểm tra kỹ hơn ở các lần mua sau "
            "để việc xử lý được nhanh hơn."
        )

    return (
        f"{opener} Trường hợp này shop cần nhận trách nhiệm vì đơn hàng có vấn đề từ phía giao hàng hoặc chất lượng. "
        "Bên mình sẽ hỗ trợ kiểm tra đơn ngay và xử lý theo hướng phù hợp như gửi bù, đổi lại hàng hoặc hoàn tiền phần bị lỗi/thiếu. "
        "Bạn giúp mình gửi mã đơn và hình ảnh thực tế để mình xử lý nhanh cho bạn nhé."
    )


async def rewrite_with_llm(client, user_text: str, assistant_text: str, group: int, reason: str, index: int) -> str:
    feedback = ""
    for attempt in range(3):
        prompt = {
            "group": group,
            "reason": reason,
            "user_message": user_text,
            "old_assistant": assistant_text,
            "feedback_from_validator": feedback,
        }
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=420,
        )
        try:
            parsed = extract_json_payload(raw)
            candidate = str(parsed["assistant_reply"]).strip()
        except Exception as exc:
            feedback = f"Lần trước không trả đúng JSON: {type(exc).__name__}: {exc}"
            continue

        is_valid, validation_reason = validate_group_response(group, candidate)
        if is_valid:
            return candidate
        feedback = f"Lần trước câu trả lời bị loại vì: {validation_reason}. Hãy viết lại đúng quy tắc."

    return fallback_rewrite(group, user_text, index)


async def process_record(client, record, index, rewrite_shop_fault_when_bad: bool):
    user_text, assistant_text = extract_pair(record)

    group, reason = detect_group_rule_based(user_text)
    if group is None:
        group, reason = await classify_with_llm(client, user_text, assistant_text)

    # Always rewrite if it has banned phrases, even if it's group 3
    if group == 3 and not response_has_banned_style(assistant_text):
        rewritten = assistant_text
    else:
        rewritten = await rewrite_with_llm(client, user_text, assistant_text, group, reason, index)

    is_valid, validation_reason = validate_group_response(group, rewritten)
    if group != 3 and not is_valid:
        rewritten = fallback_rewrite(group, user_text, index)
        is_valid, validation_reason = validate_group_response(group, rewritten)
        if not is_valid:
            raise ValueError(f"Fallback rewrite still invalid: {validation_reason}")

    return build_chatml_record(
        user_text=user_text,
        assistant_text=rewritten,
        metadata={
            "source": "clean_old_data",
            "source_index": index,
            "group": group,
            "rewrite_reason": reason,
        },
    )


async def main():
    parser = argparse.ArgumentParser(description="Step 1: Clean and relabel legacy CSKH data.")
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=Path("data/old_data.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/clean_old_data.jsonl"))
    parser.add_argument("--failed-output", type=Path, default=Path("data/clean_old_data.failed.jsonl"))
    parser.add_argument("--resume", action="store_true", help="Skip rows already written to output.")
    args = parser.parse_args()

    api_key, base_url, model, inf_type, v_project, v_location = resolve_api_settings(args)
    records = load_jsonl(args.input)
    completed_indices = set()

    if args.resume and args.output.exists():
        for row in load_jsonl_best_effort(args.output):
            if isinstance(row, dict) and "source_index" in row:
                completed_indices.add(int(row["source_index"]))

    pending_records = [(idx, rec) for idx, rec in enumerate(records) if idx not in completed_indices]
    write_mode = "a" if args.resume and args.output.exists() else "w"
    write_lock = asyncio.Lock()

    async with AsyncInferenceClient(
        model,
        api_key,
        base_url,
        inf_type,
        args.timeout_seconds,
        args.max_concurrency,
        args.requests_per_minute,
        vertex_project=v_project,
        vertex_location=v_location,
    ) as client:
        progress = tqdm(total=len(records), desc="Cleansing", initial=len(completed_indices))

        async def append_jsonl(path: Path, row: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with write_lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if write_mode == "w":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.failed_output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("", encoding="utf-8")
            args.failed_output.write_text("", encoding="utf-8")

        async def worker(idx, rec):
            try:
                cleaned = await process_record(client, rec, idx, False)
                await append_jsonl(args.output, cleaned)
            except Exception as exc:
                await append_jsonl(args.failed_output, {
                    "index": idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "record": rec,
                })
            finally:
                progress.update(1)

        await asyncio.gather(*(worker(i, r) for i, r in pending_records))
        progress.close()

    final_cleaned = load_jsonl_best_effort(args.output)
    final_failed = load_jsonl_best_effort(args.failed_output) if args.failed_output.exists() else []
    if final_failed:
        print(f"Saved {len(final_failed)} failed records to {args.failed_output}")
    print(f"Saved {len(final_cleaned)} cleaned records to {args.output}")
    print(f"Resume state: completed={len(completed_indices)} pending_at_start={len(pending_records)}")


if __name__ == "__main__":
    asyncio.run(main())