from pathlib import Path

import orjson
import pyarrow.parquet as pq


class Corpus:
    def __init__(self, settings, split):
        self.settings = settings
        self.split = split
        self.paths = [Path(path) for path in settings['paths_by_split'][split]]

    def __iter__(self):
        return {'jsonl': self.jsonl, 'parquet': self.parquet}[self.settings['format']]()

    def __len__(self):
        counters = {'jsonl': lambda: sum(1 for _ in self.jsonl()),
                    'parquet': lambda: sum(pq.read_metadata(path).num_rows for path in self.paths)}
        return counters[self.settings['format']]()

    def jsonl(self):
        for path in self.paths:
            with path.open('rb') as stream:
                for line in stream:
                    row = orjson.loads(line)
                    if row['split'] == self.split:
                        yield row

    def parquet(self):
        for path in self.paths:
            with pq.ParquetFile(path) as file:
                for batch in file.iter_batches(batch_size=self.settings['batch_rows'],
                                               use_threads=self.settings['use_threads']):
                    for row in batch.to_pylist():
                        yield {**row, 'split': self.split}
