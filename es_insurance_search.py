#!/usr/bin/env python3.11
"""
平安雇主险文档 Elasticsearch 索引与搜索脚本
功能：
  1) 连接 ES (https://localhost:9200, 用户 elastic)
  2) 创建索引 insurance_docs
  3) 将 docs/ 下的所有 .txt 和 .pdf 文件索引为文档（PDF 含扫描件 OCR）
  4) 执行搜索
  5) 显示搜索结果
"""

import os
import glob
import sys
from pathlib import Path
# 确保项目目录在 sys.path 中，以便 import config
sys.path.insert(0, str(Path(__file__).parent))
from elasticsearch import Elasticsearch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.markup import escape


# ─── 配置 ───────────────────────────────────────────────
from config import ES_HOST, ES_USER, ES_PASS, OLD_INDEX_NAME
INDEX_NAME = OLD_INDEX_NAME  # 本脚本使用旧的纯文本索引
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
SEARCH_QUERY = "工伤保险和雇主险有什么区别？"

console = Console()


# ─── 工具函数 ──────────────────────────────────────────

def extract_pdf_text(filepath):
    """从 PDF 中提取文本，支持常规 PDF 和扫描件 OCR"""
    import pdfplumber

    pdf = pdfplumber.open(filepath)
    pages_text = []
    scanned_pages = []

    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(f"\n--- 第{i+1}页 ---\n{text.strip()}")
        else:
            scanned_pages.append(i)

    pdf.close()

    full_text = "\n".join(pages_text)

    # 如果有扫描页，用 OCR 识别
    if scanned_pages and not full_text.strip():
        # 整份 PDF 都是扫描件，提整个文本
        full_text = ocr_pdf(filepath)
    elif scanned_pages:
        # 部分页面是扫描件，未来可扩展逐页 OCR
        # 目前先只返回已有文本页
        import warnings
        warnings.warn(f"  ⚠️ 文件有 {len(scanned_pages)} 页扫描页未提取文本: {scanned_pages}")

    return full_text.strip()


def ocr_pdf(filepath):
    """对扫描件 PDF 做 OCR 识别"""
    import pypdfium2 as pdfium
    from rapidocr import RapidOCR

    engine = RapidOCR()
    pdf = pdfium.PdfDocument(filepath)
    all_text = []

    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2)  # 2x 提高识别率
        img = bitmap.to_pil()

        import numpy as np
        img_array = np.array(img)
        result = engine(img_array)

        # RapidOCR 新版返回 RapidOCROutput 对象，通过属性访问
        txts = result.txts
        if txts:
            page_text = "\n".join(txts)
            all_text.append(f"\n--- 第{i+1}页 ---\n{page_text}")

        bitmap.close()

    pdf.close()
    return "\n".join(all_text)


def get_title_from_filename(filename):
    """从文件名提取标题"""
    title = filename.replace(".txt", "").replace(".pdf", "")
    if title[0].isdigit() and "-" in title:
        title = title.split("-", 1)[1]
    return title


# ─── 主流程 ─────────────────────────────────────────────

def main():
    # ─── 标题 ────────────────────────────────────
    title_panel = Panel.fit(
        "[bold cyan]📄 平安雇主险文档搜索系统[/bold cyan]\n"
        "[dim]Elasticsearch 索引与检索工具[/dim]",
        border_style="cyan",
        padding=(1, 4),
    )
    console.print(title_panel)
    console.print()

    # ─── 1) 连接 ES ────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        console=console,
    ) as progress:
        task_connect = progress.add_task("Connecting to Elasticsearch ...", total=None)
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            ssl_show_warn=False,
        )

        if not es.ping():
            raise RuntimeError("Cannot connect to Elasticsearch – is it running?")

        info = es.info()
        progress.update(task_connect, description=f"[green]✅ Connected[/green]  (ES version: {info['version']['number']})")

    # 连接详情
    conn_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    conn_table.add_column(style="bold", width=10)
    conn_table.add_column()
    conn_table.add_row("Host", ES_HOST)
    conn_table.add_row("User", ES_USER)
    conn_table.add_row("Index", f"[cyan]{INDEX_NAME}[/cyan]")
    console.print(Panel(conn_table, border_style="blue", title="[bold]Connection[/bold]"))
    console.print()

    # ─── 2) 删除旧索引，重新创建 ────────────────────
    with console.status("[bold]Preparing index ...[/bold]"):
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
            console.print("   [yellow]🗑️ Deleted existing index[/yellow]")

        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "file_type": {"type": "keyword"},
                }
            }
        }

        es.indices.create(index=INDEX_NAME, body=settings)
        console.print(f"   [green]✅ Created index[/green] [cyan]'{INDEX_NAME}'[/cyan]")

    console.print()

    # ─── 3) 索引文档 ────────────────────────────────
    console.rule("[bold yellow]Indexing Documents")

    # 收集所有 .txt 和 .pdf 文件
    txt_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    all_files = txt_files + pdf_files

    if not all_files:
        raise FileNotFoundError(f"No .txt or .pdf files found in {DOCS_DIR}")

    file_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        title=f"Found [cyan]{len(all_files)}[/cyan] files in [underline]{DOCS_DIR}[/underline]",
    )
    file_table.add_column("#", style="dim", width=3)
    file_table.add_column("Filename", style="bold")
    file_table.add_column("Type", width=6)
    file_table.add_column("Status", width=40)
    file_table.add_column("ID")

    indexed_count = 0
    for idx, filepath in enumerate(all_files, start=1):
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        title = get_title_from_filename(filename)
        type_label = ext.replace(".", "").upper()

        # 提取文本内容
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            status_text = f"[green]✅ {len(content)} chars[/green]"
            _id = "—"
        elif ext == ".pdf":
            content = extract_pdf_text(filepath)
            if content:
                status_text = f"[green]✅ {len(content)} chars[/green]"
            else:
                status_text = "[yellow]⚠️ empty[/yellow]"
            _id = "—"
        else:
            continue

        doc = {
            "filename": filename,
            "title": title,
            "content": content,
            "file_type": ext.replace(".", ""),
        }

        resp = es.index(index=INDEX_NAME, document=doc)
        _id = resp["_id"]

        file_table.add_row(str(idx), filename, type_label, status_text, f"[dim]{_id}[/dim]")
        indexed_count += 1

    es.indices.refresh(index=INDEX_NAME)
    console.print(file_table)
    console.print(f"\n   [green]✅ Indexed [bold]{indexed_count}[/bold] documents successfully.[/green]")
    console.print()

    # ─── 4) 执行搜索 ────────────────────────────────
    console.rule("[bold yellow]Search")

    console.print(f"   Query: [bold white on blue] {escape(SEARCH_QUERY)} [/bold white on blue]")
    console.print()

    search_body = {
        "query": {
            "match": {
                "content": SEARCH_QUERY
            }
        },
        "highlight": {
            "fields": {
                "content": {
                    "fragment_size": 150,
                    "number_of_fragments": 2
                }
            }
        }
    }

    resp = es.search(index=INDEX_NAME, body=search_body)

    # ─── 5) 显示搜索结果 ────────────────────────────
    total = resp["hits"]["total"]["value"]

    console.print(f"   [bold]Total hits:[/bold] [cyan]{total}[/cyan]")
    console.print()

    if total == 0:
        console.print("[yellow]No results found.[/yellow]")
    else:
        for i, hit in enumerate(resp["hits"]["hits"], start=1):
            source = hit["_source"]
            score = hit["_score"]

            # 构建结果卡片
            result_lines = []

            # 元信息
            meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            meta.add_column(style="bold", width=12)
            meta.add_column()
            meta.add_row("📄 Title", f"[bold]{escape(source['title'])}[/bold]")
            meta.add_row("📁 File", escape(source['filename']))
            meta.add_row("🏷️ Type", source.get('file_type', '?'))
            meta_panel = Panel(meta, border_style="bright_blue", padding=(0, 1))
            result_lines.append(meta_panel)

            # 内容片段
            content_text = Text()
            if "highlight" in hit and "content" in hit["highlight"]:
                for frag in hit["highlight"]["content"]:
                    # 将 <em> 标签替换为 rich 样式
                    frag_styled = frag.replace("<em>", "[bold yellow on_red]").replace("</em>", "[/bold yellow on_red]")
                    content_text.append(f"\n   ... {frag_styled} ...\n")
            else:
                preview = source["content"][:300]
                content_text.append(f"\n   {escape(preview)}...\n")

            content_panel = Panel(
                content_text,
                border_style="dim",
                title=f"[bold]Result #{i}[/bold]  (score: {score:.3f})",
                title_align="left",
                padding=(1, 2),
            )
            result_lines.append(content_panel)

            # 将所有内容组合
            combined = Text.assemble(
                ("─" * 60, "dim"),
                "\n",
            )
            for rl in result_lines:
                combined.append_text(Text("\n"))

            # 用 Panel 包裹每个结果
            console.print(content_panel)
            console.print()

    # ─── 完成 ────────────────────────────────────
    console.rule(style="green")
    console.print("[bold green]🎉 All Done![/bold green]")
