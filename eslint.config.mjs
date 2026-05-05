import storybook from "eslint-plugin-storybook";
import lumaTs from "@luma-dev/eslint-plugin-luma-ts";

import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "**/*.local.*",
    "**/*.ignore.*",
  ]),
  ...storybook.configs["flat/recommended"],
  {
    plugins: {
      "@luma-dev/luma-ts": lumaTs,
    },
    rules: {
      "@luma-dev/luma-ts/require-satisfies-in-tls": "error",
      "@luma-dev/luma-ts/no-as-unknown-as": "error",
      "@luma-dev/luma-ts/no-explicit-return-is": "error",
      "@luma-dev/luma-ts/prefer-immutable": "error",
      "@luma-dev/luma-ts/no-date": "error",
    },
  },
]);

export default eslintConfig;
