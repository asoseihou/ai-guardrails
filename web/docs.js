"use strict";
// Swagger UI bootstrap, kept out of docs.html so the CSP needs no inline script.
window.ui = SwaggerUIBundle({
  url: "/openapi.json",
  dom_id: "#swagger-ui",
  deepLinking: true,
  presets: [SwaggerUIBundle.presets.apis],
  layout: "BaseLayout",
});
