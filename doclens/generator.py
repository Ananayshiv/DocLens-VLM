"""Single-page vision-language answer generation and explicit evidence labels.

Outputs are MODEL-PROPOSED and must be verified against source page. A retrieved
page is supporting context, not proof that the generated answer is correct.
"""
from __future__ import annotations
import torch
from .evidence import ground_proposal
from PIL import Image

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


class SmolVLMAnswerer:
    def __init__(self, model_id: str = MODEL_ID):
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as exc:
            raise RuntimeError("Install: pip install -e '.[vision]'") from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager").to(self.device).eval()

    @torch.inference_mode()
    def answer(self, question: str, page_image: Image.Image,
               max_new_tokens: int = 120) -> str:
        instructions = (
            "Read the attached document PAGE carefully. Answer the QUESTION using ONLY "
            "information visible in this image. Reply in the question's language when "
            "possible. Return ONLY the exact shortest answer text visibly printed "
            "on ONE line of the page, copied literally, without explanations or "
            "prefixes. If information is absent or unreadable, reply exactly: "
            "NOT_FOUND. Don't invent or paraphrase missing values. "
            f"QUESTION: {question}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instructions}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[page_image.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output[:, prompt_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def answer_on_page(question: str, page: dict, corpus_dir, backend: SmolVLMAnswerer) -> dict:
    from pathlib import Path
    with Image.open(Path(corpus_dir) / page["image"]) as image:
        response = backend.answer(question, image)
        image_size = image.size
    grounded = ground_proposal(response, page, image_size)
    return {**grounded, "source": page["source"], "page": page["page"],
            "page_id": page["page_id"], "image": page["image"],
            "model_proposal": response,
            "validation": "Literal OCR/PDF coordinate match only; not independent visual/factual verification"}
