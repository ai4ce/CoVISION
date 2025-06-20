// RenderResourceManager.h
#pragma once
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Matrix4.h>

class RenderResourceManager {
public:
    static RenderResourceManager& getInstance() {
        static RenderResourceManager instance;
        return instance;
    }

    void setDepthTexture(Magnum::GL::Texture2D& texture) {
        depthTexture_ = &texture;
    }

    Magnum::GL::Texture2D* getDepthTexture() const {
        return depthTexture_;
    }

    void setCam1ViewMatrix(const Magnum::Matrix4& matrix) {
        cam1ViewMatrix_ = matrix;
    }

    const Magnum::Matrix4& getCam1ViewMatrix() const {
        return cam1ViewMatrix_;
    }

    void setCam1ProjectionMatrix(const Magnum::Matrix4& matrix) {
        cam1ProjectionMatrix_ = matrix;
    }

    const Magnum::Matrix4& getCam1ProjectionMatrix() const {
        return cam1ProjectionMatrix_;
    }

    void setUseMask(bool useMask) {
        useMask_ = useMask;
    }

    const bool getUseMask() const {
        return useMask_;
    }

    void setBinaryImage(bool binaryImage) {
        binaryImage_ = binaryImage;
    }

    const bool getBinaryImage() const {
        return binaryImage_;
    }

    void setCam1NearPlane(float near) {
        cam1NearPlane_ = near;
    }
    float getCam1NearPlane() const {
        return cam1NearPlane_;
    }
    void setCam1FarPlane(float far) {
        cam1FarPlane_ = far;
    }
    float getCam1FarPlane() const {
        return cam1FarPlane_;
    }

    void setDepthTextureToNull() {
	depthTexture_ = nullptr;
    }

    void resetCameras() {
	cam1ViewMatrix_ = Magnum::Matrix4{Magnum::Math::IdentityInit};
	cam1ProjectionMatrix_ = Magnum::Matrix4{Magnum::Math::IdentityInit};
    }

private:
    RenderResourceManager() = default;
    ~RenderResourceManager() = default;
    RenderResourceManager(const RenderResourceManager&) = delete;
    RenderResourceManager& operator=(const RenderResourceManager&) = delete;

    Magnum::GL::Texture2D* depthTexture_ = nullptr;
    Magnum::Matrix4 cam1ViewMatrix_;
    Magnum::Matrix4 cam1ProjectionMatrix_;
    float cam1NearPlane_;
    float cam1FarPlane_;
    bool useMask_ = false;
    bool binaryImage_ = false;
};

