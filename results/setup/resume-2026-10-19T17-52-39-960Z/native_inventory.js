async function(nativeRoot, runDirectory) {
  const fs = await import('node:fs/promises');
  const crypto = await import('node:crypto');
  const started=Date.now(), entries=[], totals={};
  const relative=p=>p.slice(nativeRoot.length+1).replaceAll('\\','/');
  const excluded=p=>p===nativeRoot+'/manifests/artifact_inventory.json'||p===runDirectory||p.startsWith(runDirectory+'/');
  let queue=[nativeRoot];
  while(queue.length){
    const folders=queue.splice(0,32);
    const batches=await Promise.all(folders.map(async folder=>{try{return {folder,items:await fs.readdir(folder,{withFileTypes:true})};}catch(e){entries.push({path:relative(folder),error:e.code});return {folder,items:[]};}}));
    const files=[];
    for(const {folder,items} of batches)for(const item of items){const path=folder+'/'+item.name;if(item.name==='.git'||excluded(path))continue;if(item.isDirectory()&&!item.isSymbolicLink())queue.push(path);else files.push(path);}
    while(files.length){
      const chunk=files.splice(0,128);
      const records=await Promise.all(chunk.map(async path=>{const name=relative(path);try{const stat=await fs.lstat(path);if(stat.isSymbolicLink())return {path:name,kind:'symlink',target:await fs.readlink(path)};const group=name.split('/')[0],record={path:name,bytes:stat.size};totals[group]??={files:0,bytes:0};totals[group].files++;totals[group].bytes+=stat.size;if(stat.size<=8*1024**2&&!['.envs','.cache','models','vendor'].includes(group))record.sha256=crypto.createHash('sha256').update(await fs.readFile(path)).digest('hex');else record.hash_policy='Payload hashes/revisions are in asset manifests; runtime/cache files are inventoried by size.';return record;}catch(error){return {path:name,error:error.code};}}));
      entries.push(...records);
    }
  }
  entries.sort((a,b)=>a.path.localeCompare(b.path));
  const report={captured_at:new Date().toISOString(),scan_started_at:new Date(started).toISOString(),scan_seconds:(Date.now()-started)/1000,backend:'native Windows Node filesystem; no WSL process launches',project_root:nativeRoot,file_count:entries.length,logical_file_bytes:Object.values(totals).reduce((n,v)=>n+v.bytes,0),by_top_level:totals,excluded:['manifests/artifact_inventory.json',relative(runDirectory),'.git/','directory symlink traversal'],note:'Filesystem scan over the recorded interval; active index/run excluded. Logical sizes differ from allocated disk space. Large payload integrity is recorded in pinned asset manifests; later logs may change.',files:entries};
  await fs.writeFile(runDirectory+'/native_index.json',JSON.stringify(report,null,2)+'\n');
  await fs.writeFile(runDirectory+'/native_complete.json',JSON.stringify({file_count:report.file_count,logical_file_bytes:report.logical_file_bytes,scan_seconds:report.scan_seconds})+'\n');
  return {file_count:report.file_count,logical_file_bytes:report.logical_file_bytes,scan_seconds:report.scan_seconds};
}