# 👗 Vogue Vista

> **AI-Powered Virtual Fashion & Styling Platform**

Vogue Vista is a modern AI-powered fashion platform that helps users discover outfits, manage their digital wardrobe, and receive personalised styling recommendations. Built with a scalable architecture using **React, Node.js, Express, MongoDB, and Python (FastAPI)**, the platform combines intelligent fashion assistance with a seamless user experience.

---

## ✨ Features

- 👕 AI-powered outfit recommendations
- 👗 Personalised fashion suggestions based on user preferences
- 🎨 Colour palette & style analysis
- 👚 Virtual wardrobe management
- ❤️ Save and organise favourite outfits
- 🔍 Smart outfit search & filtering
- 📈 Fashion trend discovery
- 🤖 AI Stylist Assistant
- 📱 Fully responsive across desktop and mobile devices
- 🔐 Secure authentication using JWT
- ☁️ Cloud image storage support

---

# 🏗️ Architecture

```
                React + Tailwind
                       │
                    Axios API
                       │
              Node.js + Express
                       │
        ┌──────────────┴──────────────┐
        │                             │
   MongoDB Atlas              FastAPI (Python)
        │                             │
 User Data & Outfits       AI Recommendation Engine
```

---

# 🚀 Tech Stack

## Frontend

- React.js
- Vite
- Tailwind CSS
- React Router
- Axios
- Framer Motion
- GSAP

## Backend

- Node.js
- Express.js
- JWT Authentication
- bcrypt
- Multer
- Cloudinary

## AI Service

- FastAPI
- Python
- NumPy
- Pandas
- Scikit-learn
- Pillow
- OpenCV
- Transformers *(Future)*

## Database

- MongoDB Atlas
- Mongoose

---

# 📂 Project Structure

```
VogueVista/

├── client/                 # React Frontend
│   ├── public/
│   └── src/
│       ├── assets/
│       ├── components/
│       ├── context/
│       ├── hooks/
│       ├── layouts/
│       ├── pages/
│       ├── services/
│       └── App.jsx
│
├── server/                 # Express Backend
│   ├── config/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── utils/
│   └── server.js
│
├── ml-service/             # Python AI Service
│   ├── app.py
│   ├── recommendation.py
│   ├── color_analysis.py
│   ├── body_shape.py
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

---

# 🌟 Core Modules

### 👤 User Authentication

- Register & Login
- JWT Authentication
- Secure Password Hashing
- Profile Management

---

### 👚 Virtual Wardrobe

- Upload clothing items
- Organise wardrobe by category
- Filter by season, colour, and occasion
- Edit & delete wardrobe items

---

### 🤖 AI Stylist

Receive intelligent recommendations based on:

- Occasion
- Weather
- Favourite colours
- Body shape
- Fashion preferences
- Season

---

### 🎨 Colour Analysis

Upload an image to receive:

- Dominant colour extraction
- Matching colour combinations
- Styling suggestions

---

### 🔍 Smart Search

Search outfits using:

- Category
- Occasion
- Colour
- Season
- Brand
- Style

---

### ❤️ Wishlist

Save your favourite outfits and access them anytime.

---

# 🛣️ API Overview

## Authentication

```
POST   /api/auth/register
POST   /api/auth/login
GET    /api/auth/profile
PUT    /api/auth/profile
```

## Wardrobe

```
GET    /api/wardrobe
POST   /api/wardrobe
PUT    /api/wardrobe/:id
DELETE /api/wardrobe/:id
```

## Outfits

```
GET    /api/outfits
GET    /api/outfits/:id
```

## Recommendations

```
POST   /api/recommend
GET    /api/recommend/history
```

---

# 🤖 AI Endpoints

```
POST /predict/style
POST /predict/body-shape
POST /predict/color
POST /recommend/outfit
POST /recommend/weather
```

---

# 💾 Database Collections

## Users

```
name
email
password
preferences
bodyShape
avatar
```

## Wardrobe

```
userId
image
category
colour
season
brand
tags
```

## Outfits

```
title
category
description
images
occasion
colour
season
price
rating
```

## Wishlist

```
userId
outfitId
```

## Recommendations

```
userId
recommendedOutfits
reason
createdAt
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/VogueVista.git

cd VogueVista
```

---

## 2. Install Frontend Dependencies

```bash
cd client
npm install
```

---

## 3. Install Backend Dependencies

```bash
cd ../server
npm install
```

---

## 4. Install Python Dependencies

```bash
cd ../ml-service

pip install -r requirements.txt
```

---

## 5. Configure Environment Variables

Create a `.env` file inside the **server** directory.

```env
PORT=5000

MONGODB_URI=your_mongodb_connection

JWT_SECRET=your_secret_key

CLOUDINARY_CLOUD_NAME=

CLOUDINARY_API_KEY=

CLOUDINARY_API_SECRET=

AI_SERVICE_URL=http://localhost:8000
```

---

## 6. Start MongoDB

Use MongoDB Atlas or a local MongoDB instance.

---

## 7. Start the Backend

```bash
cd server

npm run dev
```

---

## 8. Start the AI Service

```bash
cd ml-service

uvicorn app:app --reload
```

---

## 9. Start the Frontend

```bash
cd client

npm run dev
```

---

Visit

```
http://localhost:5173
```

---

# 📸 Screenshots

| Home | Dashboard |
|------|-----------|
| Add Screenshot | Add Screenshot |

| AI Stylist | Virtual Wardrobe |
|------------|------------------|
| Add Screenshot | Add Screenshot |

| Recommendations | Wishlist |
|----------------|-----------|
| Add Screenshot | Add Screenshot |

---

# 🚧 Roadmap

- [ ] AI Outfit Recommendation Engine
- [ ] Virtual Wardrobe
- [ ] Body Shape Detection
- [ ] Skin Tone Analysis
- [ ] Weather-Based Recommendations
- [ ] Event-Based Styling
- [ ] AI Chat Stylist
- [ ] Virtual Try-On
- [ ] Fashion Trend Prediction
- [ ] Pinterest Integration
- [ ] Social Outfit Sharing
- [ ] Shopping & E-commerce Integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👥 Team

Developed with ❤️ by the **Vogue Vista Team**.

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub. It motivates us to keep building and improving!
