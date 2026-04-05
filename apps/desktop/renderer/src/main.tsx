import React from "react";
import ReactDOM from "react-dom/client";
import "./monaco-languages";
import { initializeMonacoVscode } from "./monaco-vscode";
import { App } from "./App";
import "./styles/app.css";

await initializeMonacoVscode();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <App />,
);
