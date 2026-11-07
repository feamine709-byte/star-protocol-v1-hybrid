#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
🛡️ TokenShield CLI (Laya INT8 Commercial Distribution)
==============================================================================
الأوامر المدعومة:
  - tokenshield activate <KEY> : تفعيل رخصة الاستخدام التجارية محلياً
  - tokenshield start          : تشغيل حارس التوكنز الصامت وفلتر أدوات MCP (<60ms)
==============================================================================
"""

import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime

# منع إنشاء ملفات __pycache__ للحفاظ على نقاء حزمة التوزيع
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".tokenshield")
LICENSE_FILE = os.path.join(USER_CONFIG_DIR, "license.json")


def _save_license(key: str) -> dict:
    """حفظ مفتاح التفعيل المشفر في مجلد المستخدم المحلي."""
    os.makedirs(USER_CONFIG_DIR, exist_ok=True)
    key_clean = key.strip()
    if len(key_clean) < 8:
        raise ValueError("مفتاح التفعيل قصير جداً أو غير صالح. يرجى إدخال مفتاح تجاري صحيح.")

    fingerprint = hashlib.sha256(key_clean.encode("utf-8")).hexdigest()[:16].upper()
    payload = {
        "status": "ACTIVE",
        "license_key_masked": f"{key_clean[:4]}****{key_clean[-4:]}" if len(key_clean) >= 8 else "****",
        "signature": fingerprint,
        "activated_at": datetime.now().isoformat(),
        "edition": "Commercial INT8 Pro"
    }
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _load_license() -> dict:
    """التحقق من حالة التفعيل الحالية."""
    if not os.path.exists(LICENSE_FILE):
        return None
    try:
        with open(LICENSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "ACTIVE":
            return data
    except Exception:
        pass
    return None


def cmd_activate(key: str):
    """تنفيذ أمر: tokenshield activate <KEY>"""
    print("=" * 72)
    print("🔑 TokenShield Commercial License Activation")
    print("=" * 72)
    try:
        info = _save_license(key)
        print(f"[+] تم تفعيل النسخة التجارية بنجاح!")
        print(f"    • الإصدار        : {info['edition']}")
        print(f"    • المفتاح المسجل : {info['license_key_masked']}")
        print(f"    • بصمة التوثيق   : {info['signature']}")
        print(f"    • تاريخ التفعيل  : {info['activated_at']}")
        print("=" * 72)
        print("🚀 يمكنك الآن تشغيل المحرك عبر الأمر: tokenshield start")
    except ValueError as e:
        print(f"[!] خطأ في التفعيل: {e}")
        sys.exit(1)


def cmd_start(query: str = None, benchmark: bool = False):
    """تنفيذ أمر: tokenshield start"""
    print("=" * 72)
    print("🛡️ TokenShield (Laya INT8 Engine) — Transparent Background Proxy")
    print("=" * 72)

    lic = _load_license()
    if lic:
        print(f"[+] الرخصة الفعالة : {lic['edition']} (بصمة: {lic['signature']})")
    else:
        print("[!] تنبيه: لم يتم رصد مفتاح تفعيل مسجل. يمكنك التفعيل عبر: tokenshield activate <KEY>")

    t0 = time.perf_counter()
    from laya_model import LayaEngine, CONFIDENCE_FAIL_OPEN_THRESHOLD, MAX_LATENCY_BUDGET_MS

    engine = LayaEngine(model_path=MODEL_DIR)
    boot_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    print(f"[+] حالة المحرك    : جاهز في الذاكرة العشوائية (تم الإقلاع في {boot_ms} ms)")
    print(f"[+] النموذج المكمم : model_int8.onnx (CPU Execution Provider)")
    print(f"[+] الأدوات المفهرسة: {len(engine.all_mcp_ids)} خادم MCP ({engine.total_sub_tools} دالة) + {len(engine.all_skill_ids)} مهارة")
    print(f"[+] ميثاق الحماية  : Fail-Open < {CONFIDENCE_FAIL_OPEN_THRESHOLD}% | السقف الزمني < {MAX_LATENCY_BUDGET_MS} ms")
    print("=" * 72)

    if benchmark:
        print("[*] جاري تنفيذ فحص السرعة القياسي (Benchmark)...")
        b_res = engine.benchmark(iterations=5)
        print(f"    • متوسط زمن الاستجابة : {b_res['mean_ms']} ms (الوسيط: {b_res['median_ms']} ms)")
        print(f"    • أسرع / أبطأ استجابة : {b_res['min_ms']} ms / {b_res['max_ms']} ms")
        print(f"    • زمن البحث في الذاكرة: {b_res['in_memory_lookup_mean_ms']} ms")
        print("=" * 72)
        return

    if query:
        res = engine.route_and_shield(query)
        print(f"\n📌 الطلب: {query}")
        print(f"   • وضع البروتوكول : [{res['protocol_mode']}]")
        print(f"   • التصنيف والثقة : [{res['category']}] ({res['confidence']}%)")
        print(f"   • زمن الاستجابة  : {res['latency_ms']} ms")
        print(f"   • القرار         : {res['action']}")
        print(f"   • الأدوات الممررة: {res['active_tools_list']}")
        print(f"   • الأدوات المحجوبة: {res['shielded_tools_count']}")
        print("=" * 72)
        return

    print("[*] الحارس الصامت يعمل الآن. أدخل أي طلب لاختبار التوجيه والفلترة (أو اكتب 'exit' للخروج):\n")
    try:
        while True:
            user_input = input("tokenshield> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("[+] تم إيقاف TokenShield بنجاح.")
                break
            res = engine.route_and_shield(user_input)
            print(f"  ⬅️ [{res['protocol_mode']}] الفئة: {res['category']} ({res['confidence']}%) | الزمن: {res['latency_ms']} ms")
            print(f"  ⬅️ {res['action']}")
            print(f"  ⬅️ الممرر: {res['active_tools_list']} | المحجوب: {res['shielded_tools_count']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n[+] تم إيقاف TokenShield بنجاح.")


def main():
    parser = argparse.ArgumentParser(
        prog="tokenshield",
        description="TokenShield CLI — Proactive MCP & Skill Filter powered by Laya INT8 ONNX"
    )
    subparsers = parser.add_subparsers(dest="command", help="الأوامر المتاحة")

    # الأمر 1: tokenshield activate <KEY>
    act_parser = subparsers.add_parser("activate", help="تفعيل رخصة TokenShield التجارية")
    act_parser.add_argument("key", type=str, help="مفتاح التفعيل التجاري (License Key)")

    # الأمر 2: tokenshield start
    start_parser = subparsers.add_parser("start", help="تشغيل محرك التوجيه والحجب TokenShield")
    start_parser.add_argument("-q", "--query", type=str, default=None, help="فحص طلب واحد مباشرة والخروج")
    start_parser.add_argument("-b", "--benchmark", action="store_true", help="تشغيل اختبار السرعة القياسي")

    args = parser.parse_args()

    if args.command == "activate":
        cmd_activate(args.key)
    elif args.command == "start":
        cmd_start(query=args.query, benchmark=args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
