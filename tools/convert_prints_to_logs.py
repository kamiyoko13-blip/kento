"""
convert_prints_to_logs.py

安全な print -> log_info 置換補助スクリプト。
使い方:
  python tools/convert_prints_to_logs.py path/to/file1.py [path/to/file2.py ...]

処理内容:
 - AST を使って明示的な組み込み print(...) 呼び出しを log_info(...) に置換
 - 元ファイルはタイムスタンプ付き .bak に保存
 - 置換結果を上書き保存
 - 大きなリスクのある文字列結合や複雑な式はそのままにする（明示的な print 呼び出しのみ置換）

このスクリプトは自動化の補助であり、必ず差分を確認してから本番に適用してください。
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
import time

class PrintToLogTransformer(ast.NodeTransformer):
    """Replace builtin print(...) calls with log_info(...).

    Only replaces simple calls where the function is the builtin name `print`.
    Does not touch `print` if it's been rebound (e.g. assigned to a variable).
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:  # type: ignore[override]
        # If call is to a Name 'print' (no attribute access), replace
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            # Build new call: log_info(*args, **keywords)
            new_func = ast.Name(id='log_info', ctx=ast.Load())
            new_call = ast.Call(func=new_func, args=node.args, keywords=node.keywords)
            return ast.copy_location(new_call, node)
        return self.generic_visit(node)

def convert_file(path: Path) -> bool:
    src = path.read_text(encoding='utf-8')
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[SKIP] SyntaxError parsing {path}: {e}")
        return False

    transformer = PrintToLogTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)
    except Exception:
        # ast.unparse may not preserve formatting fully; fall back to writing
        # with astor if available would be nicer, but avoid adding deps.
        print(f"[WARN] ast.unparse failed for {path}; skipping")
        return False

    if new_src == src:
        print(f"[NOP] No print() calls replaced in {path}")
        return True

    # backup
    bak = path.with_name(path.name + f'.bak.{int(time.time())}')
    path.rename(bak)
    # write new file
    path.write_text(new_src, encoding='utf-8')
    print(f"[OK] Replaced print() -> log_info() in {path}. Backup saved to {bak}")
    return True

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python convert_prints_to_logs.py path/to/file.py [...]")
        return 2
    ok_all = True
    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"[ERR] Path not found: {path}")
            ok_all = False
            continue
        try:
            ok = convert_file(path)
            ok_all = ok_all and ok
        except Exception as e:
            print(f"[ERR] Failed processing {path}: {e}")
            ok_all = False
    return 0 if ok_all else 1

if __name__ == '__main__':
<<<<<<< HEAD
    raise SystemExit(main(sys.argv))
=======
    raise SystemExit(main(sys.argv))
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
