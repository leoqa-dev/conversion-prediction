# Conversion Prediction · 用户转化预测服务

一个用于预测用户转化行为的全栈项目，后端提供机器学习预测 API，前端负责数据可视化展示。

## 技术栈

后端基于 FastAPI 构建，独立的 ml 模块封装预测模型，routers 划分接口，schemas 负责请求与响应的数据校验。前端基于 Vue 3 + Vite + ECharts + Element Plus，用于展示预测结果与数据看板。

## 测试与质量

后端配置了 pytest（pytest.ini、tests 目录及覆盖率报告）用于单元测试，并使用 ruff（ruff.toml）做代码风格检查。仓库还包含 GitHub Actions 工作流（.github/workflows），用于持续集成。

## 本地运行

后端：

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

运行测试：

```bash
cd backend
pytest
```
