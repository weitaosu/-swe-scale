// Native Windows metadata reads only; WSL computation stays in the audited driver.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

(async () => {
  const root = process.cwd().replaceAll('\\', '/');
  const directory = path.resolve(process.argv[2]).replaceAll('\\', '/');
  if (!directory.startsWith(root + '/results/native_inventory/')) {
    throw new Error('Choose the active audited native inventory directory');
  }
  const source = fs.readFileSync('results/native_inventory/20261019T224817-43e7e724/executed_indexer.js', 'utf8');
  fs.writeFileSync(directory + '/executed_indexer.js', source);
  fs.writeFileSync(directory + '/execution_source.json', JSON.stringify({
    backend: 'native Windows Node filesystem via exec_command',
    args: process.argv.slice(1),
    executed_indexer_sha256: crypto.createHash('sha256').update(source).digest('hex'),
  }, null, 2));
  const index = eval('(' + source + ')');
  console.log(JSON.stringify(await index(root, directory)));
})().catch(error => { console.error(error); process.exitCode = 1; });
