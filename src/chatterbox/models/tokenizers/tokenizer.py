import logging
import json
import unicodedata
from pathlib import Path
from typing import Optional, Union, List, Any
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# --- Special Tokens Constants ---
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Thread-safe global instances for optional dependencies (Lazy Loaded)
_kakasi = None
_dicta = None
_russian_stresser = None


class BaseTokenizer:
    """Base class for tokenizers sharing common SOT/EOT checks and decoding logic."""
    
    def __init__(self, vocab_file_path: Union[str, Path]):
        self.tokenizer: Tokenizer = Tokenizer.from_file(str(vocab_file_path))
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self) -> None:
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"{SOT} not found in vocabulary."
        assert EOT in voc, f"{EOT} not found in vocabulary."

    def text_to_tokens(self, text: str, **kwargs) -> torch.Tensor:
        text_tokens = self.encode(text, **kwargs)
        return torch.IntTensor(text_tokens).unsqueeze(0)

    def encode(self, txt: str, **kwargs) -> List[int]:
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(self, seq: Union[List[int], torch.Tensor]) -> str:
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy().tolist()
            
        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt


class EnTokenizer(BaseTokenizer):
    """Standard English tokenizer utilizing space-to-special conversion mapping."""
    
    def encode(self, txt: str, **kwargs) -> List[int]:
        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids


# --- Language Specific Normalizers ---

def is_kanji(c: str) -> bool:
    """Check if character is kanji using standard Unicode block ranges."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana using standard Unicode block ranges."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese normalization: converts kanji to hiragana; leaves katakana intact."""
    global _kakasi
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
            
        result = _kakasi.convert(text)
        out = []
        for r in result:
            inp = r.get('orig', '')
            hira = r.get('hira', '')
            
            if any(is_kanji(c) for c in inp):
                if hira and hira in ["は", "へ"]:  
                    hira = " " + hira
                out.append(hira)
            elif inp and all(is_katakana(c) for c in inp):
                out.append(inp)
            else:
                out.append(inp)
                
        normalized_text = "".join(out)
        return unicodedata.normalize('NFKD', normalized_text)
    except ImportError:
        logger.warning("pykakasi not available - Japanese text processing skipped")
        return text
    except Exception as e:
        logger.error(f"Japanese processing failure: {e}")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew normalization: infuses diacritics into plaintext Hebrew."""
    global _dicta
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        return _dicta.add_diacritics(text)
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed safely: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean normalization: decomposes compound Hangul syllables into pure Jamo strings."""
    def decompose_hangul(char: str) -> str:
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        return initial + medial + final

    return ''.join(decompose_hangul(char) for char in text).strip()


def add_russian_stress(text: str) -> str:
    """Russian normalization: infuses critical lexical stress annotations."""
    global _russian_stresser
    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser
            _russian_stresser = RussianTextStresser()
        return _russian_stresser.stress_text(text)
    except ImportError:
        logger.warning("russian_text_stresser not available - Russian stress labeling skipped")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed safely: {e}")
        return text


class ChineseCangjieConverter:
    """Converts Chinese characters to discrete Cangjie tokens for tokenization parsing."""
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()

    def _load_cangjie_mapping(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        """Fetch and populate mappings securely from the repository."""
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
                cache_dir=str(model_dir) if model_dir else None
            )
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                
            for entry in data:
                parts = entry.split("\t")
                if len(parts) < 2:
                    continue
                word, code = parts[0], parts[1]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
        except Exception as e:
            logger.error(f"Could not initialize Cangjie dynamic map repository: {e}")

    def _init_segmenter(self) -> None:
        """Safely isolate segmenter initialization bounds."""
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese word segmentation skipped")
            self.segmenter = None

    def _cangjie_encode(self, glyph: str) -> Optional[str]:
        """Safely encode a single Chinese character to structural Cangjie components."""
        code = self.word2cj.get(glyph)
        if code is None or code not in self.cj2word:
            return None
            
        try:
            index = self.cj2word[code].index(glyph)
        except ValueError:
            return None  
            
        index_str = str(index) if index > 0 else ""
        return f"{code}{index_str}"

    def __call__(self, text: str) -> str:
        """Engine-level mapping loop executing atomic string allocations."""
        if self.segmenter is not None:
            try:
                segmented_words = self.segmenter.cut(text)
                full_text = " ".join(segmented_words)
            except Exception as e:
                logger.warning(f"Segmentation processing caught exception: {e}, falling back.")
                full_text = text
        else:
            full_text = text

        output = []
        for t in full_text:
            if unicodedata.category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                
                code_parts = [f"[cj_{c}]" for c in cangjie]
                code_parts.append("[cj_.]")
                output.append("".join(code_parts))
            else:
                output.append(t)
                
        return "".join(output)


class MTLTokenizer(BaseTokenizer):
    """The Ultimate Multilingual Tokenizer (MTL) handling conditional multi-dialect parsing pipelines."""
    
    def __init__(self, vocab_file_path: Union[str, Path]):
        super().__init__(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)

    def preprocess_text(self, raw_text: str, lowercase: bool = True, nfkd_normalize: bool = True) -> str:
        """Perform unified initial text normalization passes."""
        text = raw_text
        if lowercase:
            text = text.lower()
        if nfkd_normalize:
            text = unicodedata.normalize("NFKD", text)
        return text

    def encode(self, txt: str, language_id: Optional[str] = None, lowercase: bool = True, nfkd_normalize: bool = True) -> List[int]:
        txt = self.preprocess_text(txt, lowercase=lowercase, nfkd_normalize=nfkd_normalize)

        if language_id:
            lang = language_id.lower()
            if lang in ('zh', 'chs', 'cht'):
                txt = self.cangjie_converter(txt)
            elif lang == 'ja':
                txt = hiragana_normalize(txt)
            elif lang == 'he':
                txt = add_hebrew_diacritics(txt)
            elif lang == 'ko':
                txt = korean_normalize(txt)
            elif lang == 'ru':
                txt = add_russian_stress(txt)
                
            txt = f"[{lang}]{txt}"

        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids
