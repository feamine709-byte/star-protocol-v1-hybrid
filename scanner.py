import os
import sys
import json
import re
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "installed_tools.json")

class ToolDiscoveryScanner:
    """
    فاحص ديناميكي شامل لاكتشاف جميع أدوات MCP ومهارات Agentic Skills
    في بيئات Cursor, Claude Code, و Google Antigravity.
    """
    def __init__(self, user_home: str = None):
        self.user_home = user_home or os.path.expanduser("~")
        self.appdata = os.environ.get("APPDATA", os.path.join(self.user_home, "AppData", "Roaming"))
        self.localappdata = os.environ.get("LOCALAPPDATA", os.path.join(self.user_home, "AppData", "Local"))
        
        self.discovered_mcp = {}
        self.discovered_skills = {}

    def scan_all(self) -> dict:
        """تشغيل المسح الشامل وحفظ النتائج في الكاش."""
        start_time = time.perf_counter()
        
        # 1. فحص مسارات Cursor MCP
        self._scan_cursor_configs()
        
        # 2. فحص مسارات Claude Code MCP
        self._scan_claude_configs()
        
        # 3. فحص مسارات Antigravity MCP
        self._scan_antigravity_mcp_configs()
        self._scan_antigravity_mcp_directory()
        
        # 4. فحص مسارات SKILL.md
        self._scan_skills_directories()

        # بناء الكتالوج الموحد
        catalog = self._compile_catalog()
        scan_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        catalog["metadata"] = {
            "scanned_at": datetime.now().isoformat(),
            "scan_duration_ms": scan_duration_ms,
            "total_mcp_servers": len(self.discovered_mcp),
            "total_skills": len(self.discovered_skills),
            "total_entities": len(self.discovered_mcp) + len(self.discovered_skills)
        }

        # حفظ الكاش المحلي فائق السرعة
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

        return catalog

    def _scan_cursor_configs(self):
        """فحص ملفات إعدادات Cursor MCP"""
        cursor_paths = [
            os.path.join(self.user_home, ".cursor", "mcp.json"),
            os.path.join(self.appdata, "Cursor", "User", "globalStorage", "rooveterinaryinc.roo-cline", "settings", "cline_mcp_settings.json"),
            os.path.join(self.appdata, "Cursor", "User", "settings.json"),
            os.path.join(os.getcwd(), ".cursor", "mcp.json")
        ]
        
        for p in cursor_paths:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    servers = data.get("mcpServers", {})
                    for sname, sconfig in servers.items():
                        self.discovered_mcp[sname] = {
                            "id": sname,
                            "name": sname,
                            "source": "cursor_mcp",
                            "config_path": p,
                            "command": sconfig.get("command"),
                            "args": sconfig.get("args", []),
                            "description": f"Cursor MCP Server: {sname}",
                            "sub_tools": sconfig.get("autoApprove", [])
                        }
                except Exception:
                    pass

    def _scan_claude_configs(self):
        """فحص ملفات إعدادات Claude Code MCP"""
        claude_paths = [
            os.path.join(self.user_home, ".claude.json"),
            os.path.join(self.user_home, ".claude", "config.json"),
            os.path.join(self.appdata, "Claude", "claude_desktop_config.json")
        ]
        
        for p in claude_paths:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    servers = data.get("mcpServers", {})
                    for sname, sconfig in servers.items():
                        if sname not in self.discovered_mcp:
                            self.discovered_mcp[sname] = {
                                "id": sname,
                                "name": sname,
                                "source": "claude_code",
                                "config_path": p,
                                "command": sconfig.get("command"),
                                "args": sconfig.get("args", []),
                                "description": f"Claude MCP Server: {sname}",
                                "sub_tools": []
                            }
                except Exception:
                    pass

    def _scan_antigravity_mcp_configs(self):
        """فحص ملفات Antigravity MCP Configs"""
        ag_paths = [
            os.path.join(self.user_home, ".gemini", "mcp.json"),
            os.path.join(self.user_home, ".gemini", "antigravity", "mcp_config.json"),
            os.path.join(self.user_home, ".gemini", "config", "mcp_config.json")
        ]
        
        for p in ag_paths:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    servers = data.get("mcpServers", {})
                    for sname, sconfig in servers.items():
                        if sname not in self.discovered_mcp:
                            self.discovered_mcp[sname] = {
                                "id": sname,
                                "name": sname,
                                "source": "antigravity_config",
                                "config_path": p,
                                "command": sconfig.get("command"),
                                "args": sconfig.get("args", []),
                                "description": f"Antigravity MCP Server: {sname}",
                                "sub_tools": sconfig.get("autoApprove", [])
                            }
                except Exception:
                    pass

    def _scan_antigravity_mcp_directory(self):
        """فحص مجلد خوادم MCP في Antigravity لاستخراج الأدوات وأوصافها بالتفصيل"""
        mcp_base = os.path.join(self.user_home, ".gemini", "antigravity", "mcp")
        if not os.path.isdir(mcp_base):
            return

        for sname in os.listdir(mcp_base):
            s_path = os.path.join(mcp_base, sname)
            if not os.path.isdir(s_path):
                continue

            tool_files = [f for f in os.listdir(s_path) if f.endswith(".json") and f != "package.json"]
            sub_tools = []
            descriptions = []

            for tf in tool_files:
                tool_name = tf[:-5]
                sub_tools.append(tool_name)
                try:
                    with open(os.path.join(s_path, tf), "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                        tdesc = tdata.get("description", "")
                        if tdesc and len(descriptions) < 3:
                            descriptions.append(tdesc.split(".")[0])
                except Exception:
                    pass

            instr_path = os.path.join(s_path, "instructions.md")
            instructions_preview = ""
            if os.path.exists(instr_path):
                try:
                    with open(instr_path, "r", encoding="utf-8") as f:
                        instructions_preview = f.read(500).strip().replace("\n", " ")
                except Exception:
                    pass

            # تحديد فئة السيرفر
            category = self._categorize_server(sname)
            
            summary_desc = instructions_preview or "; ".join(descriptions) or f"MCP Server providing {len(sub_tools)} tools"
            # تنظيف الوصف
            summary_desc = re.sub(r"\s+", " ", summary_desc)[:250]

            self.discovered_mcp[sname] = {
                "id": sname,
                "name": sname.replace("-", " ").title() + " MCP",
                "type": "mcp_server",
                "category": category,
                "source": "antigravity_mcp",
                "path": s_path,
                "tools_count": len(sub_tools),
                "sub_tools": sub_tools,
                "description": summary_desc,
                "keywords": self._generate_keywords(sname, sub_tools, summary_desc)
            }

    def _scan_skills_directories(self):
        """فحص مجلدات SKILL.md واستخراج بطاقات الهوية"""
        skill_roots = [
            os.path.join(self.user_home, ".gemini", "config", "skills"),
            os.path.join(self.user_home, ".gemini", "config", "plugins"),
            os.path.join(self.user_home, ".gemini", "antigravity", "builtin", "skills")
        ]

        for sroot in skill_roots:
            if not os.path.isdir(sroot):
                continue

            for root, _, files in os.walk(sroot):
                for f in files:
                    if f.lower() == "skill.md":
                        fpath = os.path.join(root, f)
                        sname, sdesc = self._parse_skill_md(fpath)
                        if sname:
                            category = self._categorize_skill(sname, sdesc)
                            self.discovered_skills[sname] = {
                                "id": sname,
                                "name": sname.replace("-", " ").replace("_", " ").title(),
                                "type": "skill",
                                "category": category,
                                "source": "gemini_skills",
                                "path": fpath,
                                "description": sdesc[:250],
                                "keywords": self._generate_keywords(sname, [], sdesc)
                            }

    @staticmethod
    def _parse_skill_md(fpath: str) -> tuple:
        """استخراج الاسم والوصف من YAML frontmatter لملف SKILL.md"""
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read(8192)
            match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            folder_name = os.path.basename(os.path.dirname(fpath))
            if match:
                fm = match.group(1)
                name_match = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
                name = name_match.group(1).strip() if name_match else folder_name
                desc_match = re.search(r"^description:\s*([|>]-?)?\s*\n(.*?)(?=\n[a-zA-Z0-9_-]+:|\Z)", fm, re.DOTALL | re.MULTILINE)
                if desc_match:
                    desc = re.sub(r"\s+", " ", desc_match.group(2).strip())
                else:
                    desc_single = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
                    desc = desc_single.group(1).strip() if desc_single else f"Skill {name}"
                return name, desc
            return folder_name, f"Skill {folder_name}"
        except Exception:
            return None, None

    @staticmethod
    def _categorize_server(sname: str) -> str:
        s = sname.lower()
        if "git" in s or "github" in s:
            return "version_control"
        if "browser" in s or "patchright" in s:
            return "browser_automation"
        if "search" in s or "perplexity" in s:
            return "web_search"
        if "data" in s or "notebook" in s:
            return "data_science"
        if "rea" in s or "reverse" in s or "token" in s:
            return "reverse_engineering"
        if "docs" in s or "prompt" in s:
            return "documentation_prompts"
        return "general_mcp"

    @staticmethod
    def _categorize_skill(sname: str, desc: str) -> str:
        text = (sname + " " + desc).lower()
        if "bigquery" in text or "dbt" in text or "dataform" in text or "sql" in text:
            return "data_engineering"
        if "firebase" in text or "firestore" in text:
            return "firebase_backend"
        if "devtools" in text or "chrome" in text or "web" in text:
            return "web_frontend"
        if "reverse" in text or "binary" in text:
            return "reverse_engineering"
        return "workflow_skills"

    @staticmethod
    def _generate_keywords(name: str, sub_tools: list, desc: str) -> list:
        tokens = set(name.replace("-", " ").replace("_", " ").lower().split())
        for st in sub_tools[:15]:
            tokens.update(st.replace("_", " ").lower().split())
        for word in re.findall(r"\b[a-zA-Z]{3,15}\b", desc.lower())[:30]:
            if word not in {"the", "and", "for", "with", "from", "that", "this", "when", "use"}:
                tokens.add(word)
        return sorted(list(tokens))

    def _compile_catalog(self) -> dict:
        return {
            "mcp_servers": self.discovered_mcp,
            "skills": self.discovered_skills
        }

if __name__ == "__main__":
    print("=" * 70)
    print("[*] تشغيل Auto-Discovery Scanner لفحص الأدوات والمهارات في النظام...")
    print("=" * 70)

    scanner = ToolDiscoveryScanner()
    catalog = scanner.scan_all()
    meta = catalog["metadata"]

    print(f"\n[+] اكتمل المسح في: {meta['scan_duration_ms']} ms")
    print(f"[+] عدد خوادم MCP المكتشفة: {meta['total_mcp_servers']}")
    print(f"[+] عدد المهارات (Skills) المكتشفة: {meta['total_skills']}")
    print(f"[+] الإجمالي الكلي للكائنات: {meta['total_entities']}")
    print(f"[+] تم حفظ الكاش في: {CACHE_FILE}\n")

    print("=" * 70)
    print("قائمة خوادم MCP المكتشفة (MCP Servers):")
    print("-" * 70)
    for s_id, s in catalog["mcp_servers"].items():
        cnt = s.get("tools_count", len(s.get("sub_tools", [])))
        print(f" • [{s_id:<24}] | الفئة: {s['category']:<18} | الأدوات: {cnt:3d} | المصدر: {s['source']}")

    print("\n" + "=" * 70)
    print("عينة من مهارات Agentic Skills المكتشفة (Sample of 56 Skills):")
    print("-" * 70)
    sample_skills = list(catalog["skills"].items())[:12]
    for sk_id, sk in sample_skills:
        print(f" • [{sk_id:<32}] | الفئة: {sk['category']:<18} | {sk['description'][:50]}...")
    print(f" ... و {len(catalog['skills']) - len(sample_skills)} مهارة إضافية مسجلة بالكامل في installed_tools.json")
    print("=" * 70)
