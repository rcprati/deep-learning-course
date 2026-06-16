/**
 * screenshot-slides.mjs
 *
 * Captura screenshots de cada slide usando a API interna do Slidev
 * (window.__slidev__.nav.go), evitando SPA routing e 404s.
 *
 * Uso: node scripts/screenshot-slides.mjs <baseUrl> <outDir>
 * Ex:  node scripts/screenshot-slides.mjs http://localhost:8080/ /tmp/slides-lec01
 */

import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const [,, baseUrl, outDir] = process.argv;

if (!baseUrl || !outDir) {
  console.error('Uso: screenshot-slides.mjs <baseUrl> <outDir>');
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
});

const page = await browser.newPage();
await page.setViewportSize({ width: 1920, height: 1080 });

// Navega para o slide 1 e aguarda o app Slidev inicializar
console.log(`Abrindo ${baseUrl}...`);
await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 30000 });

// Força body visível (caso o tema esconda inicialmente)
await page.addStyleTag({
  content: 'body,html{display:block!important;visibility:visible!important;opacity:1!important}'
}).catch(() => {});

// Aguarda window.__slidev__ estar disponível
await page.waitForFunction(
  () => window.__slidev__?.nav?.total > 0,
  { timeout: 30000 }
).catch(async () => {
  // Fallback: tenta forçar body e aguardar mais
  await page.evaluate(() => {
    document.body.style.cssText = 'display:block!important;visibility:visible!important;opacity:1!important';
  });
  await page.waitForFunction(() => window.__slidev__?.nav?.total > 0, { timeout: 15000 });
});

const total = await page.evaluate(() => window.__slidev__.nav.total);
console.log(`Total de slides: ${total}`);

for (let i = 1; i <= total; i++) {
  // Navega usando a API interna do Slidev (sem recarregar a página)
  await page.evaluate((n) => window.__slidev__.nav.go(n), i);

  // Aguarda o slide correto estar ativo
  await page.waitForFunction(
    (n) => window.__slidev__.nav.currentPage === n,
    i,
    { timeout: 10000 }
  ).catch(() => {});

  // Pequena pausa para animações e fontes
  await page.waitForTimeout(600);

  const num = String(i).padStart(3, '0');
  await page.screenshot({
    path: `${outDir}/slide-${num}.png`,
    clip: { x: 0, y: 0, width: 1920, height: 1080 }
  });

  process.stdout.write(`  ${i}/${total}\r`);
}

await browser.close();
console.log(`\nConcluído: ${total} slides em ${outDir}`);
