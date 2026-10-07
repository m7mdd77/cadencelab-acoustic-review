"""Package only explicit source, tests and reviewed dataset artifacts."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build():
    root = Path(__file__).resolve().parent
    source = [path for path in root.iterdir() if path.is_file() and
              (path.suffix in {'.py', '.mjs', '.html', '.css', '.ps1'} or path.name in
               {'README.md', 'TECHNICAL_REPORT.md', 'requirements.txt',
                'requirements-alignment.txt', 'CadenceLab-technical-report.pdf', 'LICENSE', 'demo-manifest.json'})]
    dataset = [path for path in (root / 'dataset').rglob('*') if path.is_file()
               and path.suffix in {'.wav', '.json', '.txt'}]
    files = sorted(source + dataset)
    entries = []
    output = root / 'CadenceLab-source-dataset.zip'
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            if any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(root).parts):
                raise ValueError('Private/generated directory in package')
            raw = path.read_bytes()
            archive.writestr('cadencelab/' + relative, raw)
            entries.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        manifest = {'schemaVersion': 1, 'scope': 'Local source/dataset package, not a submitted or published entry',
                    'exclusions': ['model weights', 'cache', 'virtual environment', 'credentials', 'logs'],
                    'files': entries}
        archive.writestr('cadencelab/package-manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Package integrity failed')
    print(json.dumps({'file': output.name, 'files': len(entries), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    build()
