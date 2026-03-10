#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def load_replacements(json_path: Path) -> List[Tuple[str, str]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"JSON файл не найден: {json_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ошибка JSON ({json_path}): {e}")

    pairs: List[Tuple[str, str]] = []
    seen_ci: Dict[str, str] = {}

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"Элемент #{i} не объект: {item!r}")

        original = item.get("original")
        alias = item.get("alias")

        if not isinstance(original, str) or not isinstance(alias, str):
            raise RuntimeError(f"Элемент #{i} должен содержать строки original/alias: {item!r}")

        if original == "":
            raise RuntimeError(f"Элемент #{i}: original не должен быть пустым")

        # Проверка конфликтов
        key_ci = original.casefold()
        if key_ci in seen_ci and seen_ci[key_ci] != original:
            raise RuntimeError(
                f"Конфликт original без учета регистра: {seen_ci[key_ci]!r} и {original!r}. "
                "Оставьте только один вариант."
            )
        seen_ci[key_ci] = original

        pairs.append((original, alias))

    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def build_regex(replacements: List[Tuple[str, str]], whole_words: bool = False) -> Tuple[re.Pattern, Dict[str, str]]:
    mapping_ci = {orig.casefold(): alias for orig, alias in replacements}

    escaped = [re.escape(orig) for orig, _ in replacements]
    if not escaped:
        return re.compile(r"(?!x)x"), mapping_ci

    pattern_body = "|".join(escaped)

    if whole_words:
        pattern = re.compile(rf"\b(?:{pattern_body})\b", flags=re.IGNORECASE)
    else:
        pattern = re.compile(pattern_body, flags=re.IGNORECASE)

    return pattern, mapping_ci


def replace_text(text: str, pattern: re.Pattern, mapping_ci: Dict[str, str]) -> Tuple[str, int]:
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        src = match.group(0)
        count += 1
        return mapping_ci[src.casefold()]

    new_text = pattern.sub(repl, text)
    return new_text, count


def process_file(
    file_path: Path,
    pattern: re.Pattern,
    mapping_ci: Dict[str, str],
    encoding: str = "utf-8",
    preview: bool = False,
    create_backup: bool = True,
) -> Tuple[bool, int, str]:
    try:
        original_text = file_path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return False, 0, f"SKIP (decode error, encoding={encoding})"
    except Exception as e:
        return False, 0, f"ERROR (read): {e}"

    new_text, replacements_count = replace_text(original_text, pattern, mapping_ci)

    if replacements_count == 0:
        return False, 0, "UNCHANGED"

    if preview:
        return True, replacements_count, "PREVIEW"

    try:
        if create_backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            if not backup_path.exists():
                backup_path.write_text(original_text, encoding=encoding)

        file_path.write_text(new_text, encoding=encoding)
        return True, replacements_count, "UPDATED"
    except Exception as e:
        return False, 0, f"ERROR (write): {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Рекурсивная замена слов/фраз в .txt файлах по JSON (без учета регистра).")
    parser.add_argument("root_folder", help="Корневая папка для обхода")
    parser.add_argument("json_file", help="JSON файл с заменами (массив объектов original/alias)")
    parser.add_argument("--preview", action="store_true", help="Только показать изменения, без записи")
    parser.add_argument("--no-backup", action="store_true", help="Не создавать .bak копии")
    parser.add_argument("--encoding", default="utf-8", help="Кодировка .txt файлов (по умолчанию utf-8)")
    parser.add_argument(
        "--whole-words",
        action="store_true",
        help="Заменять только по границам слов",
    )

    args = parser.parse_args()

    root = Path(args.root_folder)
    json_path = Path(args.json_file)

    if not root.exists() or not root.is_dir():
        print(f"Ошибка: папка не найдена или это не папка: {root}", file=sys.stderr)
        return 1

    try:
        replacements = load_replacements(json_path)
    except RuntimeError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    if not replacements:
        print("Список замен пустой. Нечего делать.")
        return 0

    pattern, mapping_ci = build_regex(replacements, whole_words=args.whole_words)

    txt_files = list(root.rglob("*.txt"))
    if not txt_files:
        print(f"В папке {root} не найдено .txt файлов")
        return 0

    total_files = 0
    changed_files = 0
    total_replacements = 0
    errors = 0

    print(f"Найдено .txt файлов: {len(txt_files)}")
    print(f"Режим: {'PREVIEW' if args.preview else 'WRITE'}")
    print(f"Backup: {'OFF' if args.no_backup else 'ON'}")
    print("Поиск original: БЕЗ учета регистра")
    print("-" * 80)

    for file_path in txt_files:
        total_files += 1
        changed, count, status = process_file(
            file_path=file_path,
            pattern=pattern,
            mapping_ci=mapping_ci,
            encoding=args.encoding,
            preview=args.preview,
            create_backup=not args.no_backup,
        )

        if status.startswith("ERROR"):
            errors += 1

        if changed:
            changed_files += 1
            total_replacements += count

        print(f"[{status:<10}] {file_path} | replacements={count}")

    print("-" * 80)
    print("Готово:")
    print(f"  Файлов проверено:   {total_files}")
    print(f"  Файлов изменено:    {changed_files}")
    print(f"  Всего замен:        {total_replacements}")
    print(f"  Ошибок:             {errors}")

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())