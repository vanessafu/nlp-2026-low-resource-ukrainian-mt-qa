#!/usr/bin/env python3
"""
filter_dochplt.py — Targeted cleaning pipeline for dochplt.tsv (en-uk parallel corpus)
  python scripts/filter_dochplt.py --input  data/training/MT/en-uk/dochplt.tsv --output data/training/MT/en-uk/dochplt_clean.tsv \\
      [--no-align] [--no-near-dedup] [--positive-domains-only] [--stats-dir stats/]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import polars as pl

from cleaning_pipeline import (
    DeduplicationStage,
    FilterStage,
    LangIDStage,
    SemanticAlignmentStage,
    StageStats,
    StructuralFilterStage,
    UnicodeNormStage,
)

logger = logging.getLogger("filter_dochplt")


# =============================================================================
# ADULT / NSFW CONTENT FILTER
# =============================================================================

class AdultContentFilter(FilterStage):
    """
    Rejects sentence pairs containing adult / pornographic content.
    Applies whole-word regex matching on both EN source and UK target.
    Handles common lexical variants and both Cyrillic and Latin spellings.
    """

    name = "SF_adult"

    _EN_PATTERNS: list[str] = [
        # Adult platforms and genres
        r"\bporn(?:ograph(?:y|ic)|hub|star|site|film|video|tube)?\b",
        r"\bxxx\b",
        r"\berotica?\b",
        r"\badult\s+(?:content|entertainment|video|film|website|site|material|dating|chat)\b",
        r"\b18\+\s*(?:only|content|site|material|rated)\b",
        r"\blive\s+(?:sex|cam|strip)\b",
        r"\bonlyfans\b",
        r"\bcamgirl\b",
        r"\bwebcam\s+(?:sex|show|girl|model)\b",
        # Services
        r"\bescort(?:ing|\s+service|\s+agency|\s+girl|\s+boy|\s+ad)?\b",
        r"\bprostitut(?:e|ion|ed|ing)\b",
        r"\bcall\s+girl\b",
        r"\bhooker\b",
        r"\bbrothel\b",
        r"\bsex\s+worker\b",
        r"\bsex\s+(?:shop|toy|tape|cam|chat|video|club|club|resort)\b",
        r"\bstrip(?:per|tease|club|ping)?\b",
        r"\bstriptease\b",
        # Sexual acts and content
        r"\bfetish\b",
        r"\bbdsm\b",
        r"\bswingers?\b",
        r"\bnude\b",
        r"\bnaked\b",
        r"\bexplicit\s+(?:content|material|video|image)\b",
        r"\banal\s+sex\b",
        r"\boral\s+sex\b",
        r"\bsexting\b",
        # Child exploitation (hard block)
        r"\bchild\s+(?:pornograph|porn|abuse|exploitat)\b",
        r"\bunderage\b.{0,30}\bsex\b",
        r"\blolita\b",
        # Common vulgar anatomical terms used in explicit context
        r"\bbig\s+(?:tits|boobs|ass|cock|dick)\b",
        r"\bcock\s*suck\b",
    ]

    _UK_PATTERNS: list[str] = [
        # Platforms / genres
        r"\bпорно(?:граф(?:ія|ічн)|фільм|зірка|сайт|хаб|відео|туб)?\b",
        r"\bеротика?\b",
        r"\bеротичн(?:ий|а|е|і)\b",
        r"\bдорослий\s+контент\b",
        r"\bдля\s+дорослих\b",
        r"\bконтент\s+18\+\b",
        r"\b18\+\s*(?:контент|матеріал|сайт)\b",
        r"\bвебкам(?:ерна|ерник|модел)?\b",
        r"\bкамгерл\b",
        r"\bонлифанс\b",
        # Services
        r"\bескорт(?:-послуги|и|-сервіс|\s+агентств|\s+дівчин|\s+реклам)?\b",
        r"\bпові(?:я|ї)\b",
        r"\bпроститутк(?:а|и)\b",
        r"\bбордел(?:ь|і)\b",
        r"\bсексуальн(?:а|і|ий)\s+послуг\b",
        r"\bсекс-(?:чат|шоп|камера|відео|клуб|магазин)\b",
        r"\bстриптиз(?:ерка|ер)?\b",
        r"\bнічний\s+клуб\b.{0,40}\bдівчат\b",
        # Sexual acts and content
        r"\bфетиш\b",
        r"\bбдсм\b",
        r"\bсвінгер(?:и|ський)?\b",
        r"\bгол(?:а|ий|і|е)\b.{0,20}\bфот(?:о|огра)\b",
        r"\bоголен(?:а|ий|і)\b",
        r"\bмастурбац(?:ія|ії)\b",
        r"\bорал(?:ьний\s+секс)?\b",
        r"\bанал(?:ьний\s+секс)?\b",
        r"\bінтим(?:ний|ні|на)\s+(?:послуги|знайомства|фото)\b",
        r"\bстатевий\s+акт\b",
        # Child exploitation (hard block)
        r"\bдитяч(?:а|е|і)\s+порнограф\b",
        r"\bнеповноліт(?:ній|ня|ні)\b.{0,30}\bсекс\b",
        # Russian-script adult terms (common contamination in UK corpora)
        r"\bпорнограф(?:ия|ическ)\b",
        r"\bпроститутк(?:а|и)\b",
        r"\bэскорт\b",
        r"\bэротика?\b",
    ]

    def __init__(self) -> None:
        self._en_re = re.compile("|".join(self._EN_PATTERNS), re.IGNORECASE)
        self._uk_re = re.compile("|".join(self._UK_PATTERNS), re.IGNORECASE)

    def _is_adult(self, src: str, tgt: str) -> bool:
        return bool(self._en_re.search(src) or self._uk_re.search(tgt))

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        keep = [
            not self._is_adult(s, t)
            for s, t in zip(df["source"].to_list(), df["target"].to_list())
        ]
        df = df.filter(pl.Series("_adult_keep", keep, dtype=pl.Boolean))
        return df, self.stats(n_in, df, adult_content=n_in - len(df))


# =============================================================================
# SPAM / SEO / LOW-QUALITY CONTENT FILTER
# =============================================================================

class SpamQualityFilter(FilterStage):
    """
    Rejects pairs with clear spam, SEO-stuffing, gambling, pharma-spam,
    or contact-ad signals. Also rejects structurally degenerate text
    (too many URLs, phone-number dumps).
    """

    name = "SF_spam"

    _EN_PATTERNS: list[str] = [
        # Pharma spam
        r"\b(?:buy|order|cheap|generic|discount)\s+(?:viagra|cialis|levitra|xanax|tramadol)\b",
        r"\bno\s+prescription\s+(?:needed|required)\b",
        r"\bfda[\s-]approved\s+online\b",
        # Casino / gambling spam
        r"\bonline\s+(?:casino|poker|roulette|slot|gambling|betting)\b",
        r"\bslot\s+machine\b",
        r"\bsports?\s+(?:betting|wagering)\b",
        r"\bfree\s+(?:spins|bonus|chips)\b.{0,30}\bcasino\b",
        r"\bwin\s+(?:real\s+)?money\s+(?:online|playing|at)\b",
        # Get-rich / MLM spam
        r"\bmake\s+(?:money|cash)\s+(?:fast|quick|online|from\s+home|easily)\b",
        r"\bwork\s+from\s+home\b.{0,50}\b(?:per\s+(?:day|week|month)|\$\d+)\b",
        r"\bget\s+rich\s+quick\b",
        r"\bpassive\s+income\s+stream\b",
        r"\bmlm\b",
        # SEO / link-spam
        r"\bbuy\s+(?:backlinks?|seo|traffic|followers?|likes?)\b",
        r"\bseo\s+(?:service|tool|rank|boost|expert|agenc)\b",
        r"\bhigh[\s-]?(?:da|pr)\s+backlinks?\b",
        r"\bguest\s+post(?:ing)?\s+service\b",
        # Click-bait ad patterns
        r"\bclick\s+here\s+(?:to\s+)?(?:buy|order|get|learn|download)\b",
        r"\blimited\s+time\s+offer\b",
        r"\bact\s+now\b",
        r"\bexclusive\s+deal\b",
        # Messenger ad dumps
        r"(?:whatsapp|telegram|wechat|viber)\s*[:\-]\s*\+?\d[\d\s\-]{7,}",
    ]

    _UK_PATTERNS: list[str] = [
        # Casino / gambling
        r"\bказино\s+(?:онлайн|бонус|безкоштовно|ігри|гральн)\b",
        r"\bонлайн[\s-]казино\b",
        r"\bставки\s+(?:на\s+спорт|онлайн|спортивн)\b",
        r"\bспортивн(?:і|их)\s+ставк\b",
        r"\bбукмекер\b",
        r"\bгральн(?:ий|ого|і)\s+автомат\b",
        r"\bфрі[\s-]?спін\b",
        # Pharma
        r"\bвіагра\b",
        r"\bціаліс\b",
        r"\bбез\s+рецепт(?:а|у)\b.{0,30}\bкупити\b",
        r"\bкупити\s+(?:дешево|онлайн)\s+(?:таблетк|ліки|препарат)\b",
        # Get-rich / MLM
        r"\bзаробити\s+(?:гроші\s+)?(?:в\s+інтернеті|онлайн|швидко|вдома)\b",
        r"\bпасивний\s+дохід\b",
        r"\bмережевий\s+маркетинг\b",
        r"\bмлм\b",
        # Messenger ad dumps
        r"(?:вайбер|вотсап|телеграм)\s*[:\-]\s*\+?\d[\d\s\-]{7,}",
    ]

    _RE_URL   = re.compile(r"https?://\S+|www\.\S+")
    _RE_PHONE = re.compile(r"\+\d[\d\s\-\(\)]{8,}\d")

    def __init__(self, max_urls: int = 3, max_phones: int = 2) -> None:
        self._en_re    = re.compile("|".join(self._EN_PATTERNS), re.IGNORECASE)
        self._uk_re    = re.compile("|".join(self._UK_PATTERNS), re.IGNORECASE)
        self.max_urls   = max_urls
        self.max_phones = max_phones

    def _classify(self, src: str, tgt: str) -> Optional[str]:
        if self._en_re.search(src) or self._uk_re.search(tgt):
            return "keyword_spam"
        n_url = len(self._RE_URL.findall(src)) + len(self._RE_URL.findall(tgt))
        if n_url > self.max_urls:
            return "url_heavy"
        n_ph = len(self._RE_PHONE.findall(src)) + len(self._RE_PHONE.findall(tgt))
        if n_ph > self.max_phones:
            return "phone_dump"
        return None

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in    = len(df)
        reasons: dict[str, int] = {}
        keep    = []
        for s, t in zip(df["source"].to_list(), df["target"].to_list()):
            reason = self._classify(s, t)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                keep.append(False)
            else:
                keep.append(True)
        df    = df.filter(pl.Series("_spam_keep", keep, dtype=pl.Boolean))
        n_rej = n_in - len(df)
        st    = StageStats(self.name, n_in=n_in, n_out=len(df),
                           n_rejected=n_rej, reject_reasons=reasons)
        return df, st


# =============================================================================
# DOMAIN HEURISTIC FILTER  (optional — --positive-domains-only)
# =============================================================================

class DomainHeuristicFilter(FilterStage):
    """
    Scores each pair for positive-domain signals (news, spoken/conversational,
    literary, educational/scientific) and rejects those matching none of them.

    Enabled only when --positive-domains-only is passed. The default threshold
    requires at least one positive signal on either side. Adjust --domain-min-hits
    to be more or less strict.
    """

    name = "SF_domain"

    # News / journalism signals
    _EN_NEWS = re.compile(
        r"\b(?:accord(?:ing)?\s+to|report(?:ed|ing)?|said\s+(?:in\s+a\s+)?statement"
        r"|announc(?:ed|ement)|press\s+release|official(?:ly)?|spokesman"
        r"|correspondent|breaking\s+news|wire\s+service|reuters|ap\s+news"
        r"|associated\s+press|afp\b|ministry\s+of|parliament|legislat"
        r"|journalist|editorial|op[\s-]?ed|headline)\b",
        re.IGNORECASE,
    )
    _UK_NEWS = re.compile(
        r"\b(?:повідомляє|повідомила|заявив|заявила|за\s+даними|прес-служба"
        r"|офіційно|міністерство|парламент|верховна\s+рада|президент"
        r"|журналіст|редакційн|заголовок|кореспондент|агентство\s+новин)\b",
        re.IGNORECASE,
    )

    # Spoken / conversational signals
    _EN_SPOKEN = re.compile(
        r'(?:^|\s)["""].*?["""]|'
        r"\bsaid(?:\s+he|\s+she|\s+they|\s+that)?\b|"
        r"\bask(?:ed|ing)\b|\banswer(?:ed|ing)\b|"
        r"\b(?:i\s+(?:think|believe|feel|mean|know)|"
        r"you\s+know|kind\s+of|sort\s+of|by\s+the\s+way|"
        r"well,\s+(?:i|you|we))\b",
        re.IGNORECASE,
    )
    _UK_SPOKEN = re.compile(
        r'(?:^|\s)[«»""„].*?[«»""„]|'
        r'\b(?:сказав|сказала|запитав|запитала|відповів|відповіла|'
        r'я\s+(?:думаю|вважаю|знаю|маю\s+на\s+увазі)|'
        r'знаєте|якщо\s+чесно|ну,\s+(?:я|ми|ви))\b',
        re.IGNORECASE,
    )

    # Literary / narrative signals
    _EN_LITERARY = re.compile(
        r"\b(?:chapter|novel|story|narrator|protagonist|character"
        r"|metaphor|allegory|once\s+upon|she\s+(?:looked|walked|smiled|said)"
        r"|he\s+(?:looked|walked|smiled|said)|the\s+night\s+was"
        r"|the\s+(?:sun|moon|sky|wind|rain)\s+(?:was|had|rose|fell)"
        r"|author|poem|verse|stanza|prose)\b",
        re.IGNORECASE,
    )
    _UK_LITERARY = re.compile(
        r"\b(?:розділ|роман|оповідання|повість|ліричн|герой|персонаж"
        r"|метафора|алегор|казка|байка|вона\s+(?:дивилась|йшла|усміхнулась|сказала)"
        r"|він\s+(?:дивився|йшов|усміхнувся|сказав)|ніч\s+була"
        r"|сонце\s+(?:сходило|зайшло|світило)|автор|поема|вірш|строфа|проза)\b",
        re.IGNORECASE,
    )

    # Educational / scientific signals
    _EN_EDUC = re.compile(
        r"\b(?:research|study|analysis|university|professor|scientist|experiment"
        r"|hypothesis|theory|methodology|peer[\s-]reviewed|journal|findings?"
        r"|data\s+(?:show|suggest|indicate)|according\s+to\s+(?:the\s+)?study"
        r"|published\s+in|statistic|survey|evidence)\b",
        re.IGNORECASE,
    )
    _UK_EDUC = re.compile(
        r"\b(?:дослідження|аналіз|університет|профес(?:ор|ійн)|науков"
        r"|експеримент|гіпотез|теорія|методолог|рецензован|журнал"
        r"|результати|дані\s+(?:показують|свідчать)|опублікован"
        r"|статистик|опитування|свідчення)\b",
        re.IGNORECASE,
    )

    _DOMAIN_CHECKS = [
        ("news",      _EN_NEWS,     _UK_NEWS),
        ("spoken",    _EN_SPOKEN,   _UK_SPOKEN),
        ("literary",  _EN_LITERARY, _UK_LITERARY),
        ("education", _EN_EDUC,     _UK_EDUC),
    ]

    def __init__(self, min_hits: int = 1) -> None:
        self.min_hits = min_hits

    def _domain_hits(self, src: str, tgt: str) -> int:
        hits = 0
        for _, en_re, uk_re in self._DOMAIN_CHECKS:
            if en_re.search(src) or uk_re.search(tgt):
                hits += 1
        return hits

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StageStats]:
        n_in = len(df)
        keep = [
            self._domain_hits(s, t) >= self.min_hits
            for s, t in zip(df["source"].to_list(), df["target"].to_list())
        ]
        df = df.filter(pl.Series("_dom_keep", keep, dtype=pl.Boolean))
        return df, self.stats(n_in, df, no_positive_domain=n_in - len(df))


# =============================================================================
# PIPELINE BUILDER AND RUNNER
# =============================================================================

def build_stages(args: argparse.Namespace) -> list[FilterStage]:
    stages: list[FilterStage] = [
        UnicodeNormStage(),
        StructuralFilterStage(),
        AdultContentFilter(),
        SpamQualityFilter(
            max_urls=args.max_urls,
            max_phones=args.max_phones,
        ),
    ]

    if args.positive_domains_only:
        stages.append(DomainHeuristicFilter(min_hits=args.domain_min_hits))

    stages.append(DeduplicationStage(
        exact=True,
        near=not args.no_near_dedup,
        threshold=0.90,
        batch_size=50_000,
    ))

    stages.append(LangIDStage(
        src_lang="en",
        threshold=args.lid_threshold,
        model_path=args.fasttext_model,
    ))

    if not args.no_align:
        stages.append(SemanticAlignmentStage(
            model=args.embed_model,
            sim_min=args.embed_sim,
            batch_size=args.embed_batch,
        ))

    return stages


def run(args: argparse.Namespace) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} …")
    df = pl.read_csv(
        str(input_path),
        separator="\t",
        has_header=True,
        quote_char=None,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    ).rename(lambda c: c.strip())

    # Ensure exactly source + target columns
    if "source" not in df.columns or "target" not in df.columns:
        logger.error("Expected columns 'source' and 'target'; got: %s", df.columns)
        sys.exit(1)
    df = df.select(["source", "target"]).drop_nulls()
    print(f"  {len(df):,} pairs after loading\n")

    stages = build_stages(args)

    print("── Setup ───────────────────────────────────────────────────────────")
    for stage in stages:
        logger.info("Initializing %s …", stage.name)
        stage.setup()

    print("\n── Filtering ───────────────────────────────────────────────────────")
    all_stats: list[StageStats] = []
    for stage in stages:
        df, stats = stage.apply(df)
        all_stats.append(stats)
        pct = stats.retention_rate * 100
        print(f"  [{stats.stage_name}]  {stats.n_in:,} → {stats.n_out:,}  ({pct:.1f}%)")
        for reason, count in stats.reject_reasons.items():
            print(f"       {reason}: {count:,}")

    # Final cleanup: drop null pairs, then deduplicate by source and target independently
    out = df.select([c for c in ["source", "target"] if c in df.columns])
    n_before = len(out)
    out = out.drop_nulls()
    n_after_nulls = len(out)
    out = out.unique(subset=["source"], keep="first", maintain_order=True)
    n_after_src = len(out)
    out = out.unique(subset=["target"], keep="first", maintain_order=True)
    n_final = len(out)
    nulls_removed = n_before - n_after_nulls
    src_dupes_removed = n_after_nulls - n_after_src
    tgt_dupes_removed = n_after_src - n_final
    if nulls_removed or src_dupes_removed or tgt_dupes_removed:
        print(f"\n── Post-pipeline dedup ─────────────────────────────────────────────")
        if nulls_removed:
            print(f"  null rows removed:    {nulls_removed:,}")
        if src_dupes_removed:
            print(f"  duplicate source removed: {src_dupes_removed:,}")
        if tgt_dupes_removed:
            print(f"  duplicate target removed: {tgt_dupes_removed:,}")

    if output_path.suffix == ".parquet":
        out.write_parquet(str(output_path), compression="zstd", compression_level=3)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source\ttarget\n")
            for row in out.iter_rows():
                f.write(f"{row[0]}\t{row[1]}\n")

    print(f"\nWrote {len(out):,} pairs → {output_path}")

    print("\n── Summary ─────────────────────────────────────────────────────────")
    for s in all_stats:
        print(f"  {s}")
    if all_stats:
        n_in  = all_stats[0].n_in
        n_out = all_stats[-1].n_out
        print(f"\n  Total  {n_in:,} → {n_out:,}  ({n_out / max(1, n_in) * 100:.1f}% retained)")

    if args.stats_dir:
        _save_stats(all_stats, input_path, output_path, Path(args.stats_dir))


def _save_stats(
    all_stats: list[StageStats],
    input_path: Path,
    output_path: Path,
    stats_dir: Path,
) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "input":     str(input_path),
        "output":    str(output_path),
        "total_in":  all_stats[0].n_in  if all_stats else 0,
        "total_out": all_stats[-1].n_out if all_stats else 0,
        "stages": [
            {
                "stage":         s.stage_name,
                "n_in":          s.n_in,
                "n_out":         s.n_out,
                "n_rejected":    s.n_rejected,
                "retention_pct": round(s.retention_rate * 100, 2),
                "reasons":       s.reject_reasons,
            }
            for s in all_stats
        ],
    }
    out_file = stats_dir / f"{input_path.stem}_filter_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  stats → {out_file}")


# =============================================================================
# CLI
# =============================================================================

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_DEFAULT_IN  = _REPO_ROOT / "data/training/MT/en-uk/dochplt.tsv"
_DEFAULT_OUT = _REPO_ROOT / "data/training/MT/en-uk/dochplt_clean.tsv"
_DEFAULT_STATS = _REPO_ROOT / "stats"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter dochplt.tsv: adult/spam removal + dedup + LID + alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    io = p.add_argument_group("I/O")
    io.add_argument("--input",  default=str(_DEFAULT_IN),
                    help=f"Input TSV path (default: {_DEFAULT_IN})")
    io.add_argument("--output", default=str(_DEFAULT_OUT),
                    help=f"Output TSV path (default: {_DEFAULT_OUT})")
    io.add_argument("--stats-dir", default=str(_DEFAULT_STATS),
                    help=f"Write JSON stats here (default: {_DEFAULT_STATS})")

    filt = p.add_argument_group("Content filtering")
    filt.add_argument("--max-urls",   type=int, default=3,
                      help="Reject pairs with more than N URLs (default 3)")
    filt.add_argument("--max-phones", type=int, default=2,
                      help="Reject pairs with more than N phone numbers (default 2)")
    filt.add_argument("--positive-domains-only", action="store_true",
                      help="Keep only pairs matching news / spoken / literary / "
                           "educational signals (more aggressive, may remove ~50%% of data)")
    filt.add_argument("--domain-min-hits", type=int, default=1,
                      help="Number of positive-domain signals required per pair (default 1)")

    dedup = p.add_argument_group("Deduplication")
    dedup.add_argument("--no-near-dedup", action="store_true",
                       help="Skip MinHash near-deduplication (faster; keeps near-dupes)")

    lid = p.add_argument_group("Language identification")
    lid.add_argument("--lid-threshold", type=float, default=0.80,
                     help="fasttext confidence threshold (default 0.80)")
    lid.add_argument("--fasttext-model", default=None,
                     help="Path to lid.176.bin (auto-downloaded if omitted)")

    align = p.add_argument_group("Semantic alignment (requires CUDA)")
    align.add_argument("--no-align", action="store_true",
                       help="Skip semantic alignment stage (no GPU needed)")
    align.add_argument("--embed-model", default="intfloat/multilingual-e5-small",
                       help="SentenceTransformer model name")
    align.add_argument("--embed-sim",   type=float, default=0.60,
                       help="Minimum cosine similarity threshold (default 0.60)")
    align.add_argument("--embed-batch", type=int,   default=512,
                       help="Encoding batch size (default 512)")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)
