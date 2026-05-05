@AGENTS.md

## Architecture

- Effect.ts をコアライブラリとして使用
- イミュータブルな設計を基本とする
- Effect の tagged class パターン（`Data.TaggedClass`, `Schema.TaggedClass`）でドメインモデルを定義する
- サービスは `Context.Tag` + `Layer` で構成し、依存注入する
- エラーは tagged union (`Data.TaggedError`) で型安全に扱う
- `interface` キーワードは使わない。代わりに `type` を使う

## 言語

- コード中のコメント、JSDoc、UIテキスト、コミットメッセージはすべて日本語で書く
- 専門用語には英語を併記する（例: 勾配降下法 (gradient descent)、逆伝播 (backpropagation)）
- 変数名・関数名・型名は英語のまま

## エラーハンドリング

- 前提条件が満たされていない場合に無音で早期リターン (`if (!x) return`) してはならない
- 呼び出し側で disabled にする等で防ぎつつ、万一到達した場合は `throw new Error(...)` で明示的に失敗させる
- 「何も起きない」が最悪のバグ — 必ず失敗を可視化する

## Workflow

- 各ステップごとに commit, push, CI確認を行う
