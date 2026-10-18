// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // The old question board and feedback form are now one page, /live/. Old links and
  // QR codes (which pointed at /feedback/?session=…) land there.
  redirects: {
    '/questions': '/live/',
    '/feedback': '/live/',
  },
});
