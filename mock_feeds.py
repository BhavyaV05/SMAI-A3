"""
mock_feeds.py — Local testing shim for Phase 1
================================================
Provides fetch_articles() with the identical signature as rss_parser.py
but returns hard-coded sample articles instead of hitting the network.

Usage in app.py:
    # During development / when feeds are blocked:
    from mock_feeds import fetch_articles

    # Switch to real RSS when live:
    from rss_parser import fetch_articles
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

# Realistic-looking sample articles spread across sources
_MOCK_DATA = [
    {
        "title": "OpenAI's GPT-5 Brings Real-Time Voice and Vision to ChatGPT",
        "description": (
            "OpenAI has announced GPT-5, its most capable model yet, "
            "featuring real-time voice conversation and native image understanding. "
            "The release signals a major step toward multimodal AI assistants "
            "that can reason across text, audio and visual inputs simultaneously."
        ),
        "url": "https://techcrunch.com/2026/04/25/openai-gpt5-voice-vision/",
        "source": "TechCrunch",
        "days_ago": 0,
    },
    {
        "title": "Apple's M4 Ultra Chip Sets New Benchmark Records in Pro Apps",
        "description": (
            "The newly released M4 Ultra chip in Apple's Mac Pro demolishes "
            "previous performance records in Final Cut Pro and Xcode. "
            "Developers are reporting 40% faster compile times compared to M3."
        ),
        "url": "https://www.theverge.com/2026/04/24/apple-m4-ultra-benchmark",
        "source": "The Verge",
        "days_ago": 1,
    },
    {
        "title": "India's Startup Ecosystem Crosses $200B Valuation Milestone",
        "description": (
            "India's tech startup ecosystem has collectively surpassed $200 billion "
            "in total valuation for the first time. The fintech and SaaS sectors "
            "led growth, with Bengaluru retaining its position as the top hub."
        ),
        "url": "https://yourstory.com/2026/04/india-startup-200b-valuation",
        "source": "YourStory",
        "days_ago": 1,
    },
    {
        "title": "Google's Gemini 2.0 Flash Achieves Human-Level Coding on SWE-Bench",
        "description": (
            "Gemini 2.0 Flash has scored 72.3% on the SWE-Bench coding evaluation, "
            "the first model to exceed human-level performance on this benchmark. "
            "Google says it plans to integrate the model into Android Studio."
        ),
        "url": "https://techcrunch.com/2026/04/23/gemini-flash-swe-bench/",
        "source": "TechCrunch",
        "days_ago": 2,
    },
    {
        "title": "Meta Releases Open-Source Llama 4 with 405B Parameters",
        "description": (
            "Meta AI has open-sourced Llama 4, a 405-billion-parameter model "
            "available under a permissive research licence. Early tests show "
            "it outperforms GPT-4 on several reasoning benchmarks."
        ),
        "url": "https://www.theverge.com/2026/04/22/meta-llama-4-open-source",
        "source": "The Verge",
        "days_ago": 3,
    },
    {
        "title": "Razorpay Acquires Malaysian Fintech Curlec for ₹800 Crore",
        "description": (
            "Indian payments giant Razorpay has completed its acquisition of "
            "Malaysian fintech startup Curlec, marking its first major international "
            "expansion. The deal values Curlec at approximately ₹800 crore."
        ),
        "url": "https://yourstory.com/2026/04/razorpay-curlec-acquisition",
        "source": "YourStory",
        "days_ago": 3,
    },
    {
        "title": "NVIDIA H200 GPU Supply Constraints Ease as New TSMC Fabs Come Online",
        "description": (
            "NVIDIA's H200 GPUs are becoming more widely available as TSMC's "
            "Arizona and Japan fabs ramp production. Cloud providers expect "
            "prices to drop by up to 30% over the next quarter."
        ),
        "url": "https://techcrunch.com/2026/04/21/nvidia-h200-supply/",
        "source": "TechCrunch",
        "days_ago": 4,
    },
    {
        "title": "Figma Launches AI-Powered 'Make Design' Feature to Rival Webflow",
        "description": (
            "Figma's new 'Make Design' mode uses generative AI to turn text prompts "
            "directly into production-ready component libraries. The feature "
            "threatens to disrupt low-code design tools like Webflow and Framer."
        ),
        "url": "https://www.theverge.com/2026/04/20/figma-make-design-ai",
        "source": "The Verge",
        "days_ago": 5,
    },
    {
        "title": "Zepto Raises $400M Series G at $5B Valuation to Expand Dark Stores",
        "description": (
            "Quick-commerce platform Zepto has closed a $400 million Series G round, "
            "valuing the company at $5 billion. The capital will fund 200 new "
            "dark stores across tier-2 Indian cities."
        ),
        "url": "https://yourstory.com/2026/04/zepto-series-g-400m/",
        "source": "YourStory",
        "days_ago": 5,
    },
    {
        "title": "Microsoft Embeds Copilot Deeply into Windows 12 Kernel",
        "description": (
            "Windows 12 will ship with Copilot integrated at the OS level, "
            "giving the AI assistant access to file system, clipboard, and "
            "app APIs. Privacy advocates have raised concerns about data access."
        ),
        "url": "https://techcrunch.com/2026/04/19/microsoft-copilot-windows12/",
        "source": "TechCrunch",
        "days_ago": 6,
    },
    {
        "title": "Tesla FSD v13 Passes 10 Million Mile Autonomous Driving Mark",
        "description": (
            "Tesla's Full Self-Driving v13 software has now logged over 10 million "
            "miles of fully autonomous driving in North America with zero at-fault "
            "accidents reported. Regulators are reviewing a national rollout waiver."
        ),
        "url": "https://www.theverge.com/2026/04/18/tesla-fsd-v13-10m-miles",
        "source": "The Verge",
        "days_ago": 7,
    },
    {
        "title": "Ola Electric Files for IPO; Targets ₹10,000 Crore Listing",
        "description": (
            "Ola Electric has filed its draft red herring prospectus with SEBI, "
            "seeking to raise ₹10,000 crore in what would be India's largest "
            "EV company IPO. The listing is expected in Q3 2026."
        ),
        "url": "https://yourstory.com/2026/04/ola-electric-ipo-sebi/",
        "source": "YourStory",
        "days_ago": 8,
    },
]


def fetch_articles(
    feeds: Optional[dict] = None,   # accepted but ignored
    max_articles: int = 30,
) -> list[dict]:
    """
    Return mock articles in the same format as rss_parser.fetch_articles().
    Articles are sorted newest-first and capped at max_articles.
    """
    now = datetime.now(timezone.utc)
    articles = []

    for item in _MOCK_DATA:
        published = now - timedelta(days=item["days_ago"], hours=4)
        articles.append({
            "title":       item["title"],
            "description": item["description"],
            "url":         item["url"],
            "source":      item["source"],
            "published":   published,
            "category":    None,
            "confidence":  None,
            "summary":     None,
        })

    # Sort newest-first (matches rss_parser behaviour)
    articles.sort(key=lambda a: a["published"], reverse=True)
    return articles[:max_articles]


# ─── CLI preview ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("T9.3 — Phase 1 MOCK DATA preview")
    print("=" * 60)

    for i, a in enumerate(fetch_articles(), 1):
        print(f"\n[{i:02d}] {a['title']}")
        print(f"     Source : {a['source']}")
        print(f"     Date   : {a['published'].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"     URL    : {a['url']}")
        print(f"     Desc   : {a['description'][:80]}...")
