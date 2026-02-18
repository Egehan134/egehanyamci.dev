/*
  src/data/projects.js
*/
export const projects = [
  {
    year: '2026',
    title: 'E-Commerce Analytics API',
    description: 'An end-to-end data science system handling customer segmentation.',
    detailedDescription: 'An end-to-end data science system handling customer segmentation, churn prediction, and CLV forecasting using FastAPI and Scikit-learn. Built with a focus on OOP and sustainable architecture.',
    tags: ['Python', 'FastAPI', 'Scikit-learn', 'Docker', 'PostgreSQL'],
    links: [
      { text: 'Article', href: '/posts/building-an-end-to-end-e-commerce-analytics-api' },
      { text: 'Source', href: 'https://github.com/Egehan134/OnlineRetailMachineLearningProject' }
    ]
  },
  {
    year: '2025',
    title: 'Project Management Tool',
    description: 'A robust web-based project and task management application built with Spring Boot.',
    detailedDescription: 'A comprehensive management system developed to practice modern Java web technologies. It features server-side rendering with Thymeleaf, dynamic project-specific task allocation, and an H2 in-memory database for rapid development. The architecture follows the MVC pattern, utilizing Spring Data JPA and Hibernate for efficient object-relational mapping.',
    tags: ['Java 17', 'Spring Boot', 'Spring Data JPA', 'Thymeleaf', 'H2 Database', 'Bootstrap', 'Maven'],
    links: [
      { text: 'Source', href: 'https://github.com/Egehan134/project_management_tool' }
    ]
  },
  {
    year: '2025',
    title: 'WebSocket Chat Application',
    description: 'A real-time chat implementation using WebSocket protocol, designed to demonstrate core networking concepts.',
    detailedDescription: 'A full-stack communication platform featuring instant messaging, user presence notifications, and a custom JSON message protocol for event handling. The project implements a centralized messaging architecture with Node.js, managing socket lifecycles (open, message, close) and ensuring secure connection validation. Deployed with a decoupled architecture on Render and Cloudflare Pages.',
    tags: ['Node.js', 'WebSocket', 'JavaScript', 'JSON', 'Render', 'Cloudflare'],
    links: [
      { text: 'Source', href: 'https://github.com/Egehan134/WebSocket-Chat-Application' }
    ]
  },
  {
    year: '2025',
    title: 'Credit Card Fraud Detection',
    description: 'A machine learning system designed to identify fraudulent transactions using advanced classification algorithms.',
    detailedDescription: 'This project demonstrates a robust pipeline for credit card fraud detection, specifically addressing the challenge of imbalanced datasets through sub-sampling and hyperparameter tuning. It evaluates multiple models—including Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, and Random Forests—using GridSearchCV for optimization. Performance is validated through cross-validation and detailed metrics like precision-recall and F1-score to ensure reliability.',
    tags: ['Python', 'Scikit-learn', 'Pandas', 'Numpy', 'Matplotlib', 'Jupyter Notebook'],
    links: [
      { text: 'Source', href: 'https://github.com/Egehan134/credit-fraud-detection' }
    ]
  }
];