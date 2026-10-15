// Rasterize the generated SVG figures for visual QA; no browser dependency.
const fs = require('fs');
const path = require('path');
const sharp = require(process.env.SEPARATION_SHARP_MODULE || 'sharp');

(async () => {
  const dir = path.resolve('outputs/separation_baseline/cases');
  const names = fs.readdirSync(dir).filter(name => name.endsWith('.svg'));
  for (const name of names) {
    await sharp(path.join(dir, name), {density: 144})
      .png().toFile(path.join(dir, name.replace('.svg', '.png')));
  }
  console.log(`Rendered ${names.length} case figures to PNG.`);
})().catch(error => { console.error(error); process.exit(1); });
